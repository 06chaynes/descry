"""PHP adapter backed by scip-php (davidrjenni/scip-php).

Install globally via Composer:

    composer global require davidrjenni/scip-php

(As of v0.0.2 the package's ``google/protobuf`` dep hits a known
Composer security advisory; use ``composer global config audit.ignore
'[\"PKSA-tcfz-w4fm-hhk9\"]' --json`` to allow the install for
development use.)

Invocation: ``scip-php`` at the project root. Requires
``composer.json``/``composer.lock``/``vendor/`` populated. No
``--output`` flag, so the adapter uses ``output_mode="rename"`` to
move the emitted ``index.scip`` into the descry cache.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from descry.scip.adapter import (
    AdapterConfig,
    CommandSpec,
    DiscoveredProject,
    register,
)
from descry.scip.adapters.typescript import parse_backtick_descriptors

logger = logging.getLogger(__name__)

_SCIP_PHP_GLOB_BUG_CACHE: dict[str, bool] = {}


def _scip_php_has_glob_bug() -> bool:
    """Return True when the installed scip-php still has the glob→PCRE
    crash bug.

    Detection is by **source inspection**, not version pinning, so
    this auto-clears as soon as upstream fixes the bug in any future
    release. We resolve the ``scip-php`` binary, walk to
    ``src/Composer/Composer.php`` (the file that constructs the bad
    regex), and look for the canonical buggy pattern:

        $exclusionRegex = '{(' . implode('|', $exclusions) . ')}';

    with ``$exclusions`` assigned directly from the raw composer
    exclude-from-classmap list (no glob→PCRE translation). When
    upstream adds a translation step — a ``str_replace('**/'`` call,
    a ``fnmatch`` → PCRE helper, or a known class-map-generator glob
    helper like ``GlobMatch`` / ``Glob::toRegex`` — the detector
    returns False and the pre-check becomes inert.

    The result is cached per-binary-path for the lifetime of the
    process to avoid repeated filesystem reads.
    """
    binary = shutil.which("scip-php")
    if binary is None:
        return False
    if binary in _SCIP_PHP_GLOB_BUG_CACHE:
        return _SCIP_PHP_GLOB_BUG_CACHE[binary]

    try:
        real_bin = Path(binary).resolve()
    except (OSError, RuntimeError):
        _SCIP_PHP_GLOB_BUG_CACHE[binary] = False
        return False

    composer_src = real_bin.parent.parent / "src" / "Composer" / "Composer.php"
    if not composer_src.exists():
        # If we can't inspect the source, fall through to the pre-check
        # (safe default — we'd rather skip scip-php than crash it).
        _SCIP_PHP_GLOB_BUG_CACHE[binary] = True
        return True

    try:
        text = composer_src.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        _SCIP_PHP_GLOB_BUG_CACHE[binary] = True
        return True

    # Focus the inspection on the specific function that builds the
    # bad regex (`loadProjectFiles`). Generic function names like
    # `preg_quote` appear elsewhere in the file for unrelated
    # namespace-quoting work and would false-positive a full-file
    # search.
    region_start = text.find("loadProjectFiles")
    region_end = text.find("\n    }", region_start) if region_start >= 0 else -1
    region = (
        text[region_start:region_end]
        if region_start >= 0 and region_end > region_start
        else ""
    )
    if not region:
        _SCIP_PHP_GLOB_BUG_CACHE[binary] = True
        return True

    # Markers that upstream has added a glob→PCRE translation INSIDE
    # loadProjectFiles. Any one of these indicates the bug has been
    # fixed; its presence causes us to trust scip-php normally.
    fix_markers = (
        "str_replace('**/",
        'str_replace("**/',
        "fnmatch(",
        "globToRegex",
        "glob_to_regex",
        "GlobMatch",
        "Glob::toRegex",
        "preg_quote",
        "escapeExclusion",
        "escape_exclusion",
    )
    buggy = not any(marker in region for marker in fix_markers)
    _SCIP_PHP_GLOB_BUG_CACHE[binary] = buggy
    return buggy


def _has_php_sources(pkg_dir: Path) -> bool:
    return any(pkg_dir.rglob("*.php"))


def _scip_php_incompatibility_reason(pkg_dir: Path) -> str | None:
    """Return a human-readable reason if scip-php v0.0.2 will crash
    on this package, or None if likely compatible.

    scip-php v0.0.2 has two known hard-crash patterns we can detect
    cheaply by reading composer.json:

    1. ``autoload.exclude-from-classmap`` entries containing ``**/``.
       scip-php builds a PCRE alternation ``{(entry1|entry2)}`` from
       these raw strings; a ``**`` adjacent to a group start (``(**``)
       is parsed as a (*VERB) backtracking-control and throws
       ``PcreException``. Symfony and many modern Composer packages use
       ``**/Tests/`` / ``**/bin/`` entries.

    (A second blocker — PHP 8.4 property-hook syntax tripping
    scip-php's older php-parser — can't be detected without a full
    PHP parse, so we leave that to surface as a runtime error.)
    """
    composer_json = pkg_dir / "composer.json"
    if not composer_json.exists():
        return None
    try:
        import json as _json

        data = _json.loads(composer_json.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    def _walk_autoload(section: object) -> list[str]:
        if not isinstance(section, dict):
            return []
        excludes = section.get("exclude-from-classmap") or []
        return [e for e in excludes if isinstance(e, str)]

    excludes: list[str] = []
    excludes.extend(_walk_autoload(data.get("autoload")))
    excludes.extend(_walk_autoload(data.get("autoload-dev")))
    for bad in excludes:
        if "**/" in bad or bad.startswith("**"):
            return (
                f"composer.json autoload.exclude-from-classmap contains "
                f"glob pattern {bad!r} which crashes scip-php v0.0.2 "
                f"(composer glob -> invalid PCRE)"
            )
    return None


class PhpAdapter:
    """scip-php — PHP symbols via static analysis."""

    name = "php"
    scheme = "scip-php"
    binary = "scip-php"
    extensions = (".php",)

    def discover(self, root: Path, excluded_dirs: set[str]) -> list[DiscoveredProject]:
        """Return one DiscoveredProject per top-level Composer package.

        A PHP project is a directory containing ``composer.json``. The
        adapter does not require ``vendor/`` during discovery because
        that's a runtime prerequisite for scip-php, not for descry's
        file walk. scip-php will surface its own error if vendor is
        missing at index time.
        """
        seen: set[str] = set()
        projects: list[DiscoveredProject] = []

        for marker in root.glob("*/composer.json"):
            pkg_dir = marker.parent
            if pkg_dir.name.startswith("."):
                continue
            if pkg_dir.name in excluded_dirs:
                continue
            if pkg_dir.name in seen:
                continue
            if not _has_php_sources(pkg_dir):
                continue
            seen.add(pkg_dir.name)
            projects.append(
                DiscoveredProject(
                    name=pkg_dir.name,
                    root=pkg_dir,
                    language=self.name,
                )
            )

        if not projects:
            if (root / "composer.json").exists() and _has_php_sources(root):
                projects.append(
                    DiscoveredProject(
                        name=root.name,
                        root=root,
                        language=self.name,
                    )
                )

        projects.sort(key=lambda p: p.name)
        return projects

    def build_command(
        self,
        project: DiscoveredProject,
        _out_path: Path,
        config: AdapterConfig,
    ) -> CommandSpec:
        """scip-php has no subcommands or --output flag — just run it.

        We pre-check composer.json for patterns known to crash
        scip-php v0.0.2 (the composer glob → invalid PCRE bug) and
        raise early with a clear reason. The cache runner catches
        exceptions from build_command and logs a warning; the descry
        parser still indexes the PHP sources regardless, so the
        project degrades to regex-only resolution instead of producing
        a confusing PCRE crash trace.

        Future-proofing: the pre-check runs only when the installed
        scip-php's source still contains the buggy pattern. Once
        upstream fixes the glob→PCRE translation in
        ``src/Composer/Composer.php``, ``_scip_php_has_glob_bug()``
        returns False and the pre-check is skipped automatically —
        no user intervention, no config flags.
        """
        if _scip_php_has_glob_bug():
            reason = _scip_php_incompatibility_reason(project.root)
            if reason is not None:
                logger.warning(
                    f"SCIP: skipping scip-php for {project.name}: {reason}. "
                    f"Falling back to regex-only resolution for this package."
                )
                raise RuntimeError(f"scip-php incompatibility: {reason}")

        argv: list[str] = [self.binary]
        argv.extend(config.extra_args)
        return CommandSpec(argv=argv, cwd=project.root, output_mode="rename")

    def parse_descriptors(self, raw: str) -> list[str]:
        """Delegate to the backtick parser.

        scip-php's descriptor format is not publicly documented. The
        shared backtick helper handles both backtick-wrapped and
        path-style descriptors, so it's a safe default; verify on a
        real index during the smoke test and swap if needed.
        """
        return parse_backtick_descriptors(raw)


register(PhpAdapter())
