"""Per-project SCIP generation with incremental caching.

This module manages SCIP index generation across multiple languages with smart
caching to avoid regenerating indices for unchanged projects. Language support
is registry-driven: each adapter in `descry.scip.adapter.ADAPTERS` contributes
its own discovery + command-building logic, and this module loops over the
registry rather than hardcoding per-language branches.

Currently registered adapters (see `descry.scip.adapters`):
- Rust crates (via rust-analyzer)
- TypeScript/JavaScript packages (via scip-typescript)
- Python packages (via scip-python)

Performance optimizations:
- Pre-warm rust-analyzer cache with parallel workers (Rust-specific)
- Run all adapters concurrently
- Dynamic worker count based on available memory
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import descry.scip.adapters  # noqa: F401 — side-effect: populate ADAPTERS registry
from descry._env import safe_env
from descry.scip.adapter import (
    ADAPTERS,
    AdapterConfig,
    DiscoveredProject,
    LanguageAdapter,
)

if TYPE_CHECKING:
    from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class ScipCacheManager:
    """Manages per-project SCIP generation with incremental caching.

    SCIP indices are generated per-project and cached based on source file
    checksums. When a project's sources haven't changed, the cached SCIP
    index is reused.

    Cache location: {project_root}/.descry_cache/scip/
    """

    _DEFAULT_EXCLUDED_DIRS = {
        "target",
        "node_modules",
        "dist",
        "docs",
        ".git",
        "__pycache__",
        "build",
        "vendor",
    }

    def __init__(
        self,
        project_root: Path,
        excluded_dirs: set[str] | None = None,
        scip_timeout_minutes: int | None = None,
        scip_extra_args: list[str] | None = None,
        scip_skip_crates: list[str] | None = None,
        scip_toolchain: str | None = None,
    ):
        """Initialize the cache manager.

        Args:
            project_root: Root directory of the project
            excluded_dirs: Directory names to skip during discovery.
                Defaults to a standard set including target, node_modules,
                dist, docs, .git, __pycache__, build, and vendor.
            scip_timeout_minutes: Timeout in minutes for SCIP generation.
                0 means unlimited. None means use env var or default.
            scip_extra_args: Extra arguments to pass to rust-analyzer scip.
                Defaults to ["--exclude-vendored-libraries"].
            scip_skip_crates: Crate names to skip during SCIP generation.
                Useful for crates that trigger rust-analyzer bugs.
            scip_toolchain: Rust toolchain to use for rust-analyzer
                (e.g. "1.92.0"). Uses `rustup run <toolchain>` prefix.
                None means use the default rust-analyzer on PATH.
        """
        self.project_root = project_root
        self.cache_dir = project_root / ".descry_cache" / "scip"
        self.checksums_file = self.cache_dir / "checksums.json"
        self.excluded_dirs = (
            excluded_dirs if excluded_dirs is not None else self._DEFAULT_EXCLUDED_DIRS
        )
        self._scip_timeout_minutes = scip_timeout_minutes
        self._scip_extra_args = (
            scip_extra_args
            if scip_extra_args is not None
            else ["--exclude-vendored-libraries"]
        )
        self._scip_skip_crates = set(scip_skip_crates) if scip_skip_crates else set()
        self._scip_toolchain = scip_toolchain

    def _discover_for(self, lang: str) -> List[DiscoveredProject]:
        """Run discovery for one adapter by `lang` name; empty list if unknown."""
        adapter = ADAPTERS.get(lang)
        if adapter is None:
            return []
        return adapter.discover(self.project_root, self.excluded_dirs)

    def get_projects(self) -> List[Tuple[str, str]]:
        """Auto-discover all indexable projects across every registered adapter.

        Returns:
            Sorted list of (project_name, project_type) tuples where
            project_type is the adapter's `name` (e.g. "rust", "typescript",
            "python", and any future SCIP language additions).
        """
        out: list[tuple[str, str]] = []
        for adapter in ADAPTERS.values():
            for project in adapter.discover(self.project_root, self.excluded_dirs):
                out.append((project.name, adapter.name))
        return sorted(out)

    def get_rust_crates(self) -> List[str]:
        """Auto-discover Rust crates (names only) via RustAdapter.

        Returns:
            Sorted list of crate directory names.
        """
        return [p.name for p in self._discover_for("rust")]

    def get_typescript_packages(self) -> List[str]:
        """Auto-discover TypeScript/JavaScript packages (names only) via TypeScriptAdapter."""
        return [p.name for p in self._discover_for("typescript")]

    def get_python_packages(self) -> List[str]:
        """Auto-discover Python packages (names only) via PythonAdapter."""
        return [p.name for p in self._discover_for("python")]

    def needs_update(self, project: str, project_type: str = "rust") -> bool:
        """Check if project SCIP needs regeneration.

        A project needs regeneration if:
        1. No cached SCIP file exists
        2. The source checksum has changed

        Args:
            project: Name of the project to check
            project_type: Type of project ("rust" or "typescript")

        Returns:
            True if SCIP should be regenerated
        """
        scip_file = self.cache_dir / f"{project}.scip"
        if not scip_file.exists():
            return True

        checksums = self._load_checksums()
        current_hash = self._hash_project(project, project_type)
        return checksums.get(project) != current_hash

    def _update_changed_for_adapter(
        self, adapter: LanguageAdapter, parallel: bool
    ) -> Dict[str, Path]:
        """Shared body of `update_changed_<lang>` — runs one adapter.

        Discovers projects via the adapter, applies language-specific skip
        lists (currently only Rust's `scip_skip_crates`), filters by
        checksum, pre-warms where the adapter wants it (currently only
        rust-analyzer), and dispatches `_generate_scip` either serially or
        through a bounded ThreadPoolExecutor.
        """
        projects = adapter.discover(self.project_root, self.excluded_dirs)

        # Apply Rust-scoped skip list.
        if adapter.name == "rust" and self._scip_skip_crates:
            skipped = [p.name for p in projects if p.name in self._scip_skip_crates]
            if skipped:
                logger.info(f"SCIP: Skipping configured crates: {skipped}")
            projects = [p for p in projects if p.name not in self._scip_skip_crates]

        names = [p.name for p in projects]
        if not projects:
            logger.info(f"No {adapter.name} projects found in project")
            return {}

        changed = [p for p in projects if self.needs_update(p.name, adapter.name)]

        if changed:
            logger.info(
                f"SCIP: Regenerating for {len(changed)} changed "
                f"{adapter.name} project(s): {[p.name for p in changed]}"
            )
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Rust-specific: pre-warm rust-analyzer cache when we have
            # multiple crates to amortize the workspace analysis cost.
            if adapter.name == "rust" and len(changed) > 1:
                self._prime_rust_analyzer_cache()

            if parallel and len(changed) > 1:
                max_workers = self._get_max_workers(len(changed))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(
                        executor.map(
                            lambda project: self._generate_scip(adapter, project),
                            changed,
                        )
                    )
            else:
                results = [self._generate_scip(adapter, p) for p in changed]

            checksums = self._load_checksums()
            for project, success in zip(changed, results):
                if success:
                    checksums[project.name] = self._hash_project(
                        project.name, adapter.name
                    )
            self._save_checksums(checksums)
        else:
            logger.debug(f"SCIP: All {adapter.name} projects up-to-date")

        return self._get_scip_paths(names)

    def update_all(self, parallel: bool = False) -> Dict[str, Path]:
        """Update SCIP for every registered adapter concurrently.

        Each adapter's generation runs in its own worker since adapters use
        independent tools with no shared resources.
        """
        results: Dict[str, Path] = {}

        if not ADAPTERS:
            return results

        with ThreadPoolExecutor(max_workers=max(1, len(ADAPTERS))) as executor:
            futures = {
                executor.submit(
                    self._update_changed_for_adapter, adapter, parallel
                ): adapter.name
                for adapter in ADAPTERS.values()
            }
            for future in as_completed(futures):
                lang = futures[future]
                try:
                    lang_results = future.result()
                    results.update(lang_results)
                except Exception as e:
                    logger.error(f"SCIP: {lang} generation failed: {e}")

        return results

    def _adapter_config_for(self, adapter: LanguageAdapter) -> AdapterConfig:
        """Build an AdapterConfig for `adapter` from this manager's state.

        For now only Rust has per-language state (`toolchain`, `extra_args`);
        Java/Go adapters added later get their own state here.
        """
        if adapter.name == "rust":
            return AdapterConfig(
                toolchain=self._scip_toolchain,
                extra_args=tuple(self._scip_extra_args),
            )
        return AdapterConfig()

    def _generate_scip(
        self, adapter: LanguageAdapter, project: DiscoveredProject
    ) -> bool:
        """Run one adapter's SCIP generation for one project.

        Shared across every registered adapter. Timing, first-run detection,
        timeout resolution, safe_env() application, and post-hoc file rename
        (for adapters whose indexer lacks an --output flag) all live here so
        each adapter only has to describe *what* to run via `build_command`.
        """
        output_path = self.cache_dir / f"{project.name}.scip"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        timeout_seconds = self._get_timeout()
        timeout_str = (
            "unlimited" if timeout_seconds is None else f"{timeout_seconds // 60}min"
        )
        logger.info(
            f"SCIP: Generating {adapter.name} index for {project.name}... "
            f"(timeout: {timeout_str})"
        )

        config = self._adapter_config_for(adapter)
        try:
            spec = adapter.build_command(project, output_path, config)
        except Exception as e:
            logger.warning(
                f"SCIP: {adapter.name} failed to build command for {project.name}: {e}"
            )
            return False

        env = safe_env()
        if spec.env_extras:
            env = {**env, **spec.env_extras}

        try:
            result = subprocess.run(
                spec.argv,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=str(spec.cwd),
                env=env,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                f"SCIP: {adapter.name} generation timed out for {project.name}"
            )
            return False
        except FileNotFoundError:
            logger.debug(f"SCIP: {adapter.binary} binary not on PATH")
            return False
        except Exception as e:
            logger.warning(
                f"SCIP: Error generating {adapter.name} index for {project.name}: {e}"
            )
            return False

        # Post-hoc rename for adapters whose indexer lacks an --output flag
        # (scip-go contingency). The indexer is assumed to have written
        # `index.scip` in `cwd`; move it into the expected location. We
        # accept non-zero exit codes here so long as an index file was
        # actually produced — scip-ruby, scip-dotnet, and some
        # scip-typescript sub-project runs exit non-zero on typecheck
        # errors but the index is still usable for the subset of files
        # that parsed cleanly. Treating partial output as failure
        # forfeits hundreds of thousands of resolvable references.
        if spec.output_mode == "rename":
            default_output = spec.cwd / "index.scip"
            if (
                default_output.exists()
                and default_output.resolve() != output_path.resolve()
            ):
                try:
                    default_output.replace(output_path)
                except OSError as e:
                    logger.warning(
                        f"SCIP: Failed to move {default_output} to {output_path}: {e}"
                    )
                    return False

        if output_path.exists() and output_path.stat().st_size > 0:
            size_kb = output_path.stat().st_size / 1024
            if result.returncode == 0:
                logger.info(f"SCIP: Generated {project.name}.scip ({size_kb:.1f} KB)")
            else:
                logger.warning(
                    f"SCIP: {adapter.name} exited non-zero ({result.returncode}) "
                    f"for {project.name} but wrote {size_kb:.1f} KB; keeping "
                    f"partial index."
                )
            return True

        stderr_tail = (
            result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
        )
        logger.warning(
            f"SCIP: Failed to generate {adapter.name} index for {project.name} "
            f"(exit={result.returncode}, output_exists={output_path.exists()}, "
            f"stderr_len={len(result.stderr)}): {stderr_tail}"
        )
        return False

    def _get_max_workers(self, num_items: int) -> int:
        """Get number of parallel workers based on memory constraints.

        Modern systems (32GB+ RAM) can safely run 3-4 rust-analyzer instances.
        Override with DESCRY_SCIP_WORKERS environment variable.

        Args:
            num_items: Number of items to process

        Returns:
            Number of workers to use
        """
        env_workers = os.environ.get("DESCRY_SCIP_WORKERS")
        if env_workers:
            return min(int(env_workers), num_items)

        # Detect available memory and adjust
        try:
            import psutil

            mem_gb = psutil.virtual_memory().total / (1024**3)
            if mem_gb >= 32:
                return min(4, num_items)  # 4 workers for 32GB+
            elif mem_gb >= 16:
                return min(3, num_items)  # 3 workers for 16GB+
            else:
                return min(2, num_items)  # Conservative for <16GB
        except ImportError:
            # psutil not available, use safe default
            return min(3, num_items)

    def _get_prime_threads(self) -> int:
        """Get thread count for rust-analyzer cache priming.

        Uses DESCRY_PRIME_THREADS env or defaults based on available CPUs.

        Returns:
            Number of threads for cache priming
        """
        env_threads = os.environ.get("DESCRY_PRIME_THREADS")
        if env_threads:
            return int(env_threads)
        # Default: use available CPUs minus 2 (leave room for system)
        import multiprocessing

        return max(2, multiprocessing.cpu_count() - 2)

    def _prime_rust_analyzer_cache(self) -> bool:
        """Pre-warm rust-analyzer cache with parallel workers.

        Uses `rust-analyzer analysis-stats` which can utilize multiple threads
        for initial analysis, reducing subsequent SCIP generation time.

        Returns:
            True if cache priming succeeded, False otherwise
        """
        num_threads = self._get_prime_threads()
        logger.info(f"SCIP: Pre-warming rust-analyzer cache ({num_threads} threads)...")

        try:
            # Use analysis-stats to prime the cache - it analyzes the workspace
            # without generating SCIP, which can help subsequent SCIP generation
            result = subprocess.run(
                [
                    "rust-analyzer",
                    "analysis-stats",
                    "--parallel",
                    str(self.project_root),
                ],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for cache priming
                cwd=str(self.project_root),
                env={**safe_env(), "RUST_ANALYZER_THREADS": str(num_threads)},
            )
            # analysis-stats returns non-zero for warnings (like cyclic deps)
            # but still warms the cache, so we consider it a success
            if result.returncode == 0:
                logger.info("SCIP: Cache priming complete")
            else:
                # Warnings are logged but don't prevent caching
                logger.info("SCIP: Cache priming complete (with warnings)")
                logger.debug(f"SCIP: analysis-stats output: {result.stderr[:200]}")
            return True
        except subprocess.TimeoutExpired:
            logger.warning("SCIP: Cache priming timed out")
            return False
        except FileNotFoundError:
            logger.debug("SCIP: rust-analyzer not found for cache priming")
            return False
        except Exception as e:
            logger.debug(f"SCIP: Cache priming error: {e}")
            return False

    def _get_timeout(self) -> int | None:
        """Get SCIP-generation timeout in seconds.

        Precedence: ``scip_timeout_minutes`` constructor arg > the
        ``DESCRY_SCIP_TIMEOUT`` env var (minutes, or ``0``/``none``/
        ``unlimited``) > default (no timeout).
        """
        # Config value takes priority
        if self._scip_timeout_minutes is not None:
            if self._scip_timeout_minutes == 0:
                return None  # 0 means unlimited
            return self._scip_timeout_minutes * 60

        # Then check environment variable
        env_timeout = os.environ.get("DESCRY_SCIP_TIMEOUT", "").lower()
        if env_timeout in ("0", "none", "unlimited", ""):
            return None
        return int(env_timeout) * 60

    def _hash_project(self, project: str, project_type: str = "rust") -> str:
        """Hash project sources for change detection.

        Rust/TS/Python have bespoke hashers that include dep manifests so
        cross-crate type info invalidates correctly. Other adapters fall
        back to a generic source-tree hash keyed on the adapter's declared
        extensions — dep-change invalidation is weaker but still catches
        source edits.
        """
        if project_type == "rust":
            return self._hash_rust_crate(project)
        elif project_type == "typescript":
            return self._hash_typescript_package(project)
        elif project_type == "python":
            return self._hash_python_package(project)
        else:
            return self._hash_generic_adapter(project, project_type)

    def _hash_generic_adapter(self, project: str, adapter_name: str) -> str:
        """Fallback hasher for adapters without bespoke logic.

        Hashes every source file (by the adapter's declared extensions)
        under ``project_root/project`` — or the root itself when project
        equals the root basename. Dep-manifest changes without a source
        edit won't invalidate the cache; accepted trade-off until each
        adapter gets its own hasher.
        """
        adapter = ADAPTERS.get(adapter_name)
        if adapter is None:
            raise ValueError(f"Unknown project type: {adapter_name}")

        candidate = self.project_root / project
        pkg_path = candidate if candidate.is_dir() else self.project_root

        hasher = hashlib.sha256()
        hasher.update(f"adapter:{adapter_name}".encode())

        source_files: list[Path] = []
        for ext in adapter.extensions:
            source_files.extend(pkg_path.rglob(f"*{ext}"))
        source_files.sort()

        for src_file in source_files:
            rel_parts = src_file.relative_to(pkg_path).parts
            skip = False
            for part in rel_parts[:-1]:
                if part.startswith(".") or part in self.excluded_dirs:
                    skip = True
                    break
            if skip:
                continue
            try:
                hasher.update(str(Path(*rel_parts)).encode())
                hasher.update(src_file.read_bytes())
            except (OSError, ValueError):
                pass

        return hasher.hexdigest()[:16]

    def _hash_rust_crate(self, crate: str) -> str:
        """Hash Rust crate sources + toolchain for change detection.

        Hashes:
        - Cargo.toml (dependencies affect types)
        - Workspace Cargo.lock (if present) — resolved dep versions affect types
        - All .rs files in the crate
        - Pinned rust toolchain and scip extra args (so cache invalidates when
          the user changes either)
        """
        crate_path = self.project_root / crate
        hasher = hashlib.sha256()

        # Hash pinned toolchain + extra args (cache-key component).
        hasher.update(f"toolchain:{self._scip_toolchain or ''}".encode())
        hasher.update(f"extra_args:{'|'.join(self._scip_extra_args)}".encode())

        # Hash Cargo.toml
        cargo_toml = crate_path / "Cargo.toml"
        if cargo_toml.exists():
            hasher.update(cargo_toml.read_bytes())

        # Hash workspace Cargo.lock (resolved dependency versions affect types).
        cargo_lock = self.project_root / "Cargo.lock"
        if cargo_lock.exists():
            hasher.update(cargo_lock.read_bytes())

        # Hash all .rs files (sorted for determinism)
        rs_files = sorted(crate_path.rglob("*.rs"))
        for rs_file in rs_files:
            # Skip target directory
            if "target" in rs_file.parts:
                continue
            try:
                # Include relative path in hash so renames are detected
                rel_path = rs_file.relative_to(crate_path)
                hasher.update(str(rel_path).encode())
                hasher.update(rs_file.read_bytes())
            except (OSError, ValueError):
                pass

        return hasher.hexdigest()[:16]

    def _hash_typescript_package(self, package: str) -> str:
        """Hash TypeScript/JavaScript package sources for change detection.

        Hashes:
        - package.json (dependencies affect types)
        - tsconfig.json if present
        - All .ts, .tsx, .js, .jsx files in src/

        Args:
            package: Name of the package to hash

        Returns:
            16-character hex hash string
        """
        pkg_path = self.project_root / package
        hasher = hashlib.sha256()

        # Hash package.json
        package_json = pkg_path / "package.json"
        if package_json.exists():
            hasher.update(package_json.read_bytes())

        # Hash tsconfig.json if present
        tsconfig = pkg_path / "tsconfig.json"
        if tsconfig.exists():
            hasher.update(tsconfig.read_bytes())

        # Hash all source files (sorted for determinism)
        extensions = ("*.ts", "*.tsx", "*.js", "*.jsx")
        source_files = []
        for ext in extensions:
            source_files.extend(pkg_path.glob(f"src/**/{ext}"))
        source_files = sorted(source_files)

        for src_file in source_files:
            # Skip node_modules and dist directories
            if "node_modules" in src_file.parts or "dist" in src_file.parts:
                continue
            try:
                # Include relative path in hash so renames are detected
                rel_path = src_file.relative_to(pkg_path)
                hasher.update(str(rel_path).encode())
                hasher.update(src_file.read_bytes())
            except (OSError, ValueError):
                pass

        return hasher.hexdigest()[:16]

    def _hash_python_package(self, package: str) -> str:
        """Hash Python package sources for change detection.

        Hashes:
        - pyproject.toml / setup.py / setup.cfg / requirements*.txt
          (dep/version changes affect type resolution)
        - All .py files recursively under the package (excluding hidden /
          excluded dirs)

        Args:
            package: Name of the package to hash. May be a subdir name or
                the project_root basename (single-package layout).

        Returns:
            16-character hex hash string.
        """
        candidate = self.project_root / package
        if candidate.is_dir() and (
            (candidate / "pyproject.toml").exists()
            or (candidate / "setup.py").exists()
            or (candidate / "setup.cfg").exists()
        ):
            pkg_path = candidate
        else:
            pkg_path = self.project_root

        hasher = hashlib.sha256()

        for name in (
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "requirements.txt",
            "requirements-dev.txt",
            "uv.lock",
            "poetry.lock",
            "Pipfile.lock",
        ):
            marker = pkg_path / name
            if marker.exists():
                hasher.update(f"{name}:".encode())
                hasher.update(marker.read_bytes())

        # Hash every in-scope .py file, sorted for determinism.
        py_files = []
        for candidate_path in pkg_path.rglob("*.py"):
            rel = candidate_path.relative_to(pkg_path)
            parts = rel.parts
            skip = False
            for part in parts[:-1]:
                if part.startswith(".") or part in self.excluded_dirs:
                    skip = True
                    break
            if not skip:
                py_files.append(candidate_path)
        py_files.sort()
        for py_file in py_files:
            try:
                rel_path = py_file.relative_to(pkg_path)
                hasher.update(str(rel_path).encode())
                hasher.update(py_file.read_bytes())
            except (OSError, ValueError):
                pass

        return hasher.hexdigest()[:16]

    def _get_scip_paths(self, crates: List[str]) -> Dict[str, Path]:
        """Get paths to existing SCIP files.

        Args:
            crates: List of crate names

        Returns:
            Dictionary mapping crate names to SCIP file paths (only existing files)
        """
        paths = {}
        for crate in crates:
            scip_file = self.cache_dir / f"{crate}.scip"
            if scip_file.exists():
                paths[crate] = scip_file
        return paths

    def _load_checksums(self) -> Dict[str, str]:
        """Load cached checksums from disk."""
        if self.checksums_file.exists():
            try:
                with open(self.checksums_file, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_checksums(self, checksums: Dict[str, str]):
        """Save checksums to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.checksums_file, "w", encoding="utf-8") as f:
            json.dump(checksums, f, indent=2)
