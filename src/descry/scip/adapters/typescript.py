"""TypeScript/JavaScript/Svelte adapter backed by scip-typescript."""

from __future__ import annotations

import json
import re
from pathlib import Path

from descry.scip.adapter import (
    AdapterConfig,
    CommandSpec,
    DiscoveredProject,
    register,
)

_BACKTICK_DESCRIPTOR_PATTERN = re.compile(
    r"([a-zA-Z_$][a-zA-Z0-9_$]*)(\([^)]*\)|[#./\[\]])?"
)


_SOURCE_PATTERNS = ("*.ts", "*.tsx", "*.js", "*.jsx")


def _has_src(pkg_dir: Path) -> bool:
    """True if ``pkg_dir`` has TS/TSX/JS/JSX files under ``src/``."""
    return any(any(pkg_dir.glob(f"src/**/{p}")) for p in _SOURCE_PATTERNS)


def _has_ts_js_sources(pkg_dir: Path) -> bool:
    """True if ``pkg_dir`` has TS/TSX/JS/JSX files under ``src/`` (or at
    the package root for flatter layouts without ``src/``).
    """
    if _has_src(pkg_dir):
        return True
    for pattern in _SOURCE_PATTERNS:
        for candidate in pkg_dir.glob(pattern):
            if candidate.is_file():
                return True
    return False


def _parse_ts_workspace_packages(root: Path) -> list[Path]:
    """Return package directories listed in a TS/JS workspace config.

    Supports three conventions:

    - **pnpm** — ``pnpm-workspace.yaml`` with a top-level ``packages:``
      list of glob entries.
    - **npm / yarn** — root ``package.json`` with a ``workspaces`` array
      (or ``workspaces.packages`` object form).
    - **yarn berry** — same ``workspaces`` array in ``package.json``.

    Glob entries like ``"packages/*"`` are expanded against ``root``.
    Invalid / missing configs yield an empty list; entries that resolve
    outside ``root`` are dropped.
    """
    result: list[Path] = []
    seen: set[Path] = set()
    root_resolved = root.resolve()

    def _expand(entries: list) -> None:
        for entry in entries:
            if not isinstance(entry, str) or not entry:
                continue
            try:
                if "*" in entry or "?" in entry or "[" in entry:
                    matches = [p for p in root.glob(entry) if p.is_dir()]
                else:
                    matches = [root / entry]
                for m in matches:
                    if not m.resolve().is_relative_to(root_resolved):
                        continue
                    if m.is_dir() and m not in seen:
                        seen.add(m)
                        result.append(m)
            except (OSError, ValueError, NotImplementedError):
                # pathlib raises NotImplementedError for an absolute pattern.
                continue

    pnpm_yaml = root / "pnpm-workspace.yaml"
    if pnpm_yaml.is_file():
        try:
            text = pnpm_yaml.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            text = ""
        in_packages = False
        parsed_entries: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            if not line.startswith((" ", "\t")):
                stripped = line.rstrip(":").strip()
                in_packages = stripped == "packages"
                continue
            if in_packages:
                bullet = line.lstrip()
                if bullet.startswith("-"):
                    value = bullet[1:].strip().strip("'\"")
                    if value:
                        parsed_entries.append(value)
        _expand(parsed_entries)

    root_pkg_json = root / "package.json"
    if root_pkg_json.is_file():
        try:
            import json as _json

            data = _json.loads(
                root_pkg_json.read_text(encoding="utf-8", errors="ignore")
            )
        except (OSError, ValueError):
            data = None
        if isinstance(data, dict):
            ws = data.get("workspaces")
            entries: list = []
            if isinstance(ws, list):
                entries = ws
            elif isinstance(ws, dict):
                pkgs = ws.get("packages")
                if isinstance(pkgs, list):
                    entries = pkgs
            _expand(entries)

    return result


_TS_SKIP_KEYWORDS = frozenset(
    {
        "export",
        "default",
        "async",
        "function",
        "class",
        "interface",
        "type",
        "const",
        "let",
        "var",
    }
)


def parse_backtick_descriptors(raw: str) -> list[str]:
    """Parse SCIP descriptors that use backtick-wrapped file paths.

    scip-typescript and scip-python both use the same wire format: file path
    components are wrapped in backticks, and the symbol portion follows the
    final closing backtick. Extract only the symbol components.

    Examples:
        ``src/lib/api/`client.ts`/getAuthToken().`` -> ``["getAuthToken"]``
        ``src/lib/stores/`users.ts`/UsersStore#fetchUsers().``
            -> ``["UsersStore", "fetchUsers"]``
    """
    last_backtick_end = -1
    i = 0
    while i < len(raw):
        if raw[i] == "`":
            i += 1
            while i < len(raw) and raw[i] != "`":
                i += 1
            if i < len(raw):
                last_backtick_end = i
        i += 1

    if 0 <= last_backtick_end < len(raw) - 1:
        symbol_portion = raw[last_backtick_end + 1 :]
    else:
        symbol_portion = raw
    symbol_portion = symbol_portion.lstrip("/")

    names: list[str] = []
    for match in _BACKTICK_DESCRIPTOR_PATTERN.finditer(symbol_portion):
        name = match.group(1)
        suffix = match.group(2) or ""
        if not name:
            continue
        if suffix == "/":
            continue
        if suffix.startswith("["):
            continue
        if name in _TS_SKIP_KEYWORDS:
            continue
        names.append(name)
    return names


class TypeScriptAdapter:
    """scip-typescript — TypeScript/JavaScript/Svelte symbols."""

    name = "typescript"
    scheme = "scip-typescript"
    binary = "scip-typescript"
    extensions = (".ts", ".tsx", ".js", ".jsx", ".svelte")

    def discover(self, root: Path, excluded_dirs: set[str]) -> list[DiscoveredProject]:
        """Return one DiscoveredProject per TypeScript/JavaScript package.

        Discovery order:

        1. **Workspace-aware.** Parse ``pnpm-workspace.yaml`` (pnpm) and
           root ``package.json`` ``workspaces`` array (npm / yarn), expand
           glob entries like ``"packages/*"``. Used by next.js, vite,
           and most modern monorepos.
        2. **Root.** A root ``package.json`` with TS/JS sources is a single
           project: scip-typescript run there indexes the tree under it.
           A workspace root is a package too: one whose own ``src/`` holds
           sources is a project beside its members.
        3. **Top-level fallback.** ``*/package.json`` — simple multi-
           package layouts with no root package or workspace config.
        """
        projects: list[DiscoveredProject] = []
        seen_paths: set[Path] = set()

        for pkg_dir in _parse_ts_workspace_packages(root):
            if pkg_dir in seen_paths:
                continue
            if pkg_dir.name.startswith("."):
                continue
            if any(part in excluded_dirs for part in pkg_dir.relative_to(root).parts):
                continue
            if not (pkg_dir / "package.json").exists():
                continue
            if not _has_ts_js_sources(pkg_dir):
                continue
            seen_paths.add(pkg_dir)
            rel = pkg_dir.relative_to(root)
            projects.append(
                DiscoveredProject(
                    name=str(rel),
                    root=pkg_dir,
                    language=self.name,
                )
            )

        root_pkg = root / "package.json"
        if root_pkg.exists() and (
            _has_src(root) if projects else _has_ts_js_sources(root)
        ):
            # A root package indexes the tree under it.
            projects.append(
                DiscoveredProject(name=root.name, root=root, language=self.name)
            )

        if not projects:
            for package_json in root.glob("*/package.json"):
                pkg_dir = package_json.parent
                if pkg_dir.name.startswith("."):
                    continue
                if pkg_dir.name in excluded_dirs:
                    continue
                if not _has_ts_js_sources(pkg_dir):
                    continue
                seen_paths.add(pkg_dir)
                projects.append(
                    DiscoveredProject(
                        name=pkg_dir.name,
                        root=pkg_dir,
                        language=self.name,
                    )
                )

        projects.sort(key=lambda p: p.name)
        return projects

    def build_command(
        self,
        project: DiscoveredProject,
        out_path: Path,
        config: AdapterConfig,
    ) -> CommandSpec:
        """Build the `scip-typescript index` command, run from the package
        directory. A package without a ``tsconfig.json`` (a Vite or SvelteKit
        app, a workspace fixture, plain JavaScript) is indexed with one
        written beside the index that reads its TypeScript and JavaScript
        with the current standard library: ``--infer-tsconfig`` would write
        one into the package.
        """
        argv: list[str] = [self.binary, "index", "--output", str(out_path)]
        if not (project.root / "tsconfig.json").exists():
            inferred = out_path.with_suffix(".tsconfig.json")
            inferred.unlink(missing_ok=True)
            settings = {
                "compilerOptions": {"allowJs": True, "target": "esnext"},
                "include": [f"{project.root}/**/*"],
            }
            with open(inferred, "x", encoding="utf-8") as file:
                json.dump(settings, file)
            argv.append(str(inferred))
        return CommandSpec(argv=argv, cwd=project.root)

    def parse_descriptors(self, raw: str) -> list[str]:
        return parse_backtick_descriptors(raw)


register(TypeScriptAdapter())
