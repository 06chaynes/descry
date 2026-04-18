"""TypeScript/JavaScript/Svelte adapter backed by scip-typescript."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from descry.scip.adapter import (
    AdapterConfig,
    CommandSpec,
    DiscoveredProject,
    register,
)

logger = logging.getLogger(__name__)


_BACKTICK_DESCRIPTOR_PATTERN = re.compile(
    r"([a-zA-Z_$][a-zA-Z0-9_$]*)(\([^)]*\)|[#./\[\]])?"
)
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
        """Return one DiscoveredProject per top-level `package.json`.

        A package directory must contain `package.json` and at least one
        `.ts/.tsx/.js/.jsx` source file under `src/`.
        """
        projects: list[DiscoveredProject] = []
        for package_json in root.glob("*/package.json"):
            pkg_dir = package_json.parent
            if pkg_dir.name.startswith("."):
                continue
            if pkg_dir.name in excluded_dirs:
                continue
            has_ts = list(pkg_dir.glob("src/**/*.ts")) or list(
                pkg_dir.glob("src/**/*.tsx")
            )
            has_js = list(pkg_dir.glob("src/**/*.js")) or list(
                pkg_dir.glob("src/**/*.jsx")
            )
            if not (has_ts or has_js):
                continue
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
        """Build the `scip-typescript index` command.

        Runs from the package directory. Adds `--infer-tsconfig` for
        SvelteKit/Vite packages where tsconfig.json may not be at the
        package root.
        """
        argv: list[str] = [self.binary, "index", "--output", str(out_path)]

        svelte_config = project.root / "svelte.config.js"
        vite_config = project.root / "vite.config.ts"
        if svelte_config.exists() or vite_config.exists():
            argv.append("--infer-tsconfig")
            logger.debug(
                f"SCIP: Using --infer-tsconfig for SvelteKit project {project.name}"
            )

        argv.extend(config.extra_args)
        return CommandSpec(argv=argv, cwd=project.root)

    def parse_descriptors(self, raw: str) -> list[str]:
        return parse_backtick_descriptors(raw)


register(TypeScriptAdapter())
