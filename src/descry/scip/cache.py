"""Per-project SCIP generation with incremental caching.

This module manages SCIP index generation across multiple languages with smart
caching to avoid regenerating indices for unchanged projects. Language support
is registry-driven: each adapter in `descry.scip.adapter.ADAPTERS` contributes
its own discovery + command-building logic, and this module loops over the
registry rather than hardcoding per-language branches.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import descry.scip.adapters  # noqa: F401 — side-effect: populate ADAPTERS registry
from descry._files import create_sibling, open_regular
from descry.scip.adapter import (
    ADAPTERS,
    AdapterConfig,
    DiscoveredProject,
    LanguageAdapter,
    project_dir,
    resolved_binary,
)
from descry.scip.adapters.rust import crate_dir as rust_crate_dir

logger = logging.getLogger(__name__)

# A line of a JavaScript, JVM or .NET stack trace
_FRAME = re.compile(r"\s+at ")


def _writes_through(path: Path) -> bool:
    """Whether a write to `path` would land elsewhere or wait: a symlink, a
    FIFO or a device sits there."""
    try:
        mode = path.lstat().st_mode
    except OSError:
        return False
    return not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))


class ScipCacheManager:
    """Manages per-project SCIP generation with incremental caching.

    SCIP indices are generated per-project and cached based on source file
    checksums. When a project's sources haven't changed, the cached SCIP
    index is reused.

    Cache location: {cache_dir}/scip/ (default
    {project_root}/.descry_cache/scip/)
    """

    def __init__(
        self,
        project_root: Path,
        excluded_dirs: set[str],
        scip_timeout_minutes: int | None = None,
        scip_extra_args: list[str] | None = None,
        scip_skip_crates: list[str] | None = None,
        scip_toolchain: str | None = None,
        cache_dir: Path | None = None,
    ):
        """Initialize the cache manager.

        Args:
            project_root: Root directory of the project
            excluded_dirs: Directory names to skip during discovery and
                source hashing.
            scip_timeout_minutes: Timeout in minutes for SCIP generation.
                0 means unlimited. None means use env var or default.
            scip_extra_args: Extra arguments to pass to rust-analyzer scip.
                Defaults to ["--exclude-vendored-libraries"].
            scip_skip_crates: Crate names to skip during SCIP generation.
                Useful for crates that trigger rust-analyzer bugs.
            scip_toolchain: Rust toolchain to use for rust-analyzer
                (e.g. "1.92.0"). Uses `rustup run <toolchain>` prefix.
                None means use the default rust-analyzer on PATH.
            cache_dir: Descry's cache directory; SCIP files go in its
                ``scip/`` subdirectory. Defaults to
                ``{project_root}/.descry_cache``.
        """
        self.project_root = project_root
        if cache_dir is None:
            self.cache_dir = project_root / ".descry_cache" / "scip"
        else:
            # Absolute, because indexers run with the project directory as
            # cwd rather than the caller's.
            self.cache_dir = Path(cache_dir).absolute() / "scip"
        self.checksums_file = self.cache_dir / "checksums.json"
        # Guards the read-modify-write of checksums.json across adapter threads.
        self._checksums_lock = threading.Lock()
        self.excluded_dirs = excluded_dirs
        self._scip_timeout_minutes = scip_timeout_minutes
        self._scip_extra_args = (
            scip_extra_args
            if scip_extra_args is not None
            else ["--exclude-vendored-libraries"]
        )
        self._scip_skip_crates = set(scip_skip_crates) if scip_skip_crates else set()
        self._scip_toolchain = scip_toolchain

    @staticmethod
    def _project_key(adapter_name: str, project: str) -> str:
        """Key for one project in ``checksums.json`` and in result dicts.

        Project names are only unique within an adapter.
        """
        return f"{adapter_name}:{project}"

    def _index_path(self, adapter_name: str, project: str) -> Path:
        """Cache file for one project's index.

        The name is percent-encoded so that one containing ``/`` (a
        workspace-relative crate path) is still a file directly in the
        cache directory.
        """
        return self.cache_dir / f"{adapter_name}--{quote(project, safe='')}.scip"

    @staticmethod
    def _file_signature(path: Path) -> tuple[int, int, int] | None:
        """Inode, mtime and size of `path`, or None if absent.

        Differs before and after a subprocess iff it wrote the file.
        """
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (st.st_ino, st.st_mtime_ns, st.st_size)

    def needs_update(self, project: str, adapter_name: str) -> bool:
        """Check if project SCIP needs regeneration.

        A project needs regeneration if:
        1. No cached SCIP file exists
        2. The source checksum has changed

        Args:
            project: Name of the project to check
            adapter_name: Name of the adapter that discovered it

        Returns:
            True if SCIP should be regenerated
        """
        scip_file = self._index_path(adapter_name, project)
        if not scip_file.exists():
            return True

        checksums = self._load_checksums()
        current_hash = self._hash_project(project, adapter_name)
        return checksums.get(self._project_key(adapter_name, project)) != current_hash

    def _update_changed_for_adapter(self, adapter: LanguageAdapter) -> dict[str, Path]:
        """Runs one adapter: discovers its projects, applies language-specific
        skip lists (currently only Rust's `scip_skip_crates`), filters by
        checksum, pre-warms where the adapter wants it (currently only
        rust-analyzer), and runs `_generate_scip` for each changed project.
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

            results = [self._generate_scip(adapter, p) for p in changed]

            with self._checksums_lock:
                checksums = self._load_checksums()
                for project, success in zip(changed, results, strict=True):
                    if success:
                        key = self._project_key(adapter.name, project.name)
                        checksums[key] = self._hash_project(project.name, adapter.name)
                self._save_checksums(checksums)
        else:
            logger.debug(f"SCIP: All {adapter.name} projects up-to-date")

        return self._get_scip_paths(adapter.name, names)

    def update_all(self) -> dict[str, Path]:
        """Update SCIP for every registered adapter concurrently.

        Each adapter's generation runs in its own worker since adapters use
        independent tools with no shared resources. The result is keyed by
        ``<adapter>:<project>``.
        """
        results: dict[str, Path] = {}

        if not ADAPTERS:
            return results

        # Remove indexes that no registered adapter's name prefixes.
        current = tuple(f"{name}--" for name in ADAPTERS)
        for path in self.cache_dir.glob("*.scip"):
            if not path.name.startswith(current):
                path.unlink(missing_ok=True)

        with ThreadPoolExecutor(max_workers=max(1, len(ADAPTERS))) as executor:
            futures = {
                executor.submit(self._update_changed_for_adapter, adapter): adapter.name
                for adapter in ADAPTERS.values()
            }
            for future in as_completed(futures):
                lang = futures[future]
                try:
                    lang_results = future.result()
                    results.update(lang_results)
                except Exception as e:  # noqa: BLE001 — one adapter's failure must not abandon the other languages in the pool
                    logger.error(f"SCIP: {lang} generation failed: {e}")

        return results

    def _adapter_config_for(self, adapter: LanguageAdapter) -> AdapterConfig:
        """Build an AdapterConfig for `adapter` from this manager's state.

        Only Rust has per-language state (`toolchain`, `extra_args`).
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
        timeout resolution, extra env vars, and post-hoc file rename
        (for adapters whose indexer lacks an --output flag) all live here so
        each adapter only has to describe *what* to run via `build_command`.
        """
        output_path = self._index_path(adapter.name, project.name)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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
        except Exception as e:  # noqa: BLE001 — boundary over third-party adapter code in the ADAPTERS registry
            logger.warning(
                f"SCIP: {adapter.name} failed to build command for {project.name}: {e}"
            )
            return False

        env = {**os.environ, **spec.env_extras}

        # The indexer writes the output path, and in rename mode `index.scip`
        # in its working directory. Either would be written through a link
        # or FIFO there: one in the cache directory is removed, and the run
        # stops on one in the project rather than delete it.
        default_output = spec.cwd / "index.scip"
        if _writes_through(output_path):
            output_path.unlink()
        if spec.output_mode == "rename" and _writes_through(default_output):
            logger.warning(
                f"SCIP: {default_output} is not a regular file; "
                f"not running {adapter.name} for {project.name}"
            )
            return False

        # Only a file the indexer writes during this run counts as its
        # output, not an index that was already there.
        output_before = self._file_signature(output_path)
        default_before = self._file_signature(default_output)

        if not spec.argv:
            logger.warning(
                f"SCIP: {adapter.name} built an empty command for {project.name}"
            )
            return False

        # Where argv[0] is the indexer itself, run the probed binary: an
        # indexer installed under another name (scip_dart for scip-dart)
        # still runs. argv[0] is allowed to be a wrapper (`sh -c`), which
        # this leaves alone.
        argv0 = (
            resolved_binary(adapter) if spec.argv[0] == adapter.binary else spec.argv[0]
        )
        try:
            result = subprocess.run(
                [argv0, *spec.argv[1:]],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                cwd=str(spec.cwd),
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                f"SCIP: {adapter.name} generation timed out for {project.name}"
            )
            return False
        except FileNotFoundError:
            logger.debug(f"SCIP: {adapter.binary} binary not on PATH")
            return False
        except (OSError, ValueError) as e:
            logger.warning(
                f"SCIP: Error generating {adapter.name} index for {project.name}: {e}"
            )
            return False

        # Post-hoc rename for adapters whose indexer lacks an --output flag.
        # The indexer is assumed to have written `index.scip` in `cwd`; move
        # it into the expected location. We accept non-zero exit codes here so long as an index file was
        # actually produced — scip-ruby, scip-dotnet, and some
        # scip-typescript sub-project runs exit non-zero on typecheck
        # errors but the index is still usable for the subset of files
        # that parsed cleanly. Treating partial output as failure
        # forfeits hundreds of thousands of resolvable references.
        if (
            spec.output_mode == "rename"
            and self._file_signature(default_output) not in (None, default_before)
            and default_output.resolve() != output_path.resolve()
        ):
            try:
                # Unlike a rename, this also works across filesystems.
                shutil.move(default_output, output_path)
            except OSError as e:
                logger.warning(
                    f"SCIP: Failed to move {default_output} to {output_path}: {e}"
                )
                return False

        if (
            self._file_signature(output_path) not in (None, output_before)
            and output_path.stat().st_size > 0
        ):
            size_kb = output_path.stat().st_size / 1024
            if result.returncode == 0:
                logger.info(f"SCIP: Generated {output_path.name} ({size_kb:.1f} KB)")
            else:
                logger.warning(
                    f"SCIP: {adapter.name} exited non-zero ({result.returncode}) "
                    f"for {project.name} but wrote {size_kb:.1f} KB; keeping "
                    f"partial index."
                )
            return True

        # Without the stack frames that bury the error above them, and
        # blank lines. The reason is on stderr or stdout, per indexer;
        # carry whichever wrote it.
        tail = ""
        for stream in (result.stderr, result.stdout):
            said = [
                line
                for line in stream.splitlines()
                if line.strip() and not _FRAME.match(line)
            ]
            if said:
                tail = "\n".join(said)[-500:]
                break
        logger.warning(
            f"SCIP: Failed to generate {adapter.name} index for {project.name} "
            f"(exit={result.returncode}, output_exists={output_path.exists()}, "
            f"stderr_len={len(result.stderr)}): {tail}"
        )
        return False

    def _get_prime_threads(self) -> int:
        """Get thread count for rust-analyzer cache priming.

        Uses DESCRY_PRIME_THREADS env or defaults based on available CPUs.

        Returns:
            Number of threads for cache priming
        """
        env_threads = os.environ.get("DESCRY_PRIME_THREADS")
        if env_threads:
            try:
                return int(env_threads)
            except ValueError:
                logger.warning(
                    f"DESCRY_PRIME_THREADS={env_threads!r} is not an integer; ignoring"
                )
        # Default: use available CPUs minus 2 (leave room for system)
        import multiprocessing

        return max(2, multiprocessing.cpu_count() - 2)

    def _prime_rust_analyzer_cache(self) -> None:
        """Pre-warm rust-analyzer cache with parallel workers.

        Uses `rust-analyzer analysis-stats` which can utilize multiple threads
        for initial analysis, reducing subsequent SCIP generation time.
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
                encoding="utf-8",
                errors="replace",
                timeout=600,  # 10 minute timeout for cache priming
                cwd=str(self.project_root),
                env={**os.environ, "RUST_ANALYZER_THREADS": str(num_threads)},
                check=False,
            )
            # analysis-stats exits non-zero for warnings (like cyclic deps)
            # but still warms the cache.
            logger.info("SCIP: Cache priming complete")
            logger.debug(f"SCIP: analysis-stats output: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            logger.warning("SCIP: Cache priming timed out")
        except FileNotFoundError:
            logger.debug("SCIP: rust-analyzer not found for cache priming")
        except (OSError, ValueError) as e:
            logger.debug(f"SCIP: Cache priming error: {e}")

    def _get_timeout(self) -> int | None:
        """Get SCIP-generation timeout in seconds.

        Precedence: the ``DESCRY_SCIP_TIMEOUT`` env var (minutes, or
        ``0``/``none``/``unlimited``) > ``scip_timeout_minutes`` constructor
        arg (``0`` = unlimited) > no timeout.
        """
        env_timeout = os.environ.get("DESCRY_SCIP_TIMEOUT", "").lower()
        if env_timeout in ("0", "none", "unlimited"):
            return None
        if env_timeout:
            try:
                return int(env_timeout) * 60
            except ValueError:
                logger.warning(
                    f"DESCRY_SCIP_TIMEOUT={env_timeout!r} is not an integer; ignoring"
                )

        # Anything but a number is ignored.
        if isinstance(self._scip_timeout_minutes, (int, float)):
            return self._scip_timeout_minutes * 60 or None
        return None

    def _hash_project(self, project: str, adapter_name: str) -> str:
        """Hash project sources for change detection.

        Rust/TS/Python have bespoke hashers that include dep manifests so
        cross-crate type info invalidates correctly. Other adapters fall
        back to a generic source-tree hash keyed on the adapter's declared
        extensions — dep-change invalidation is weaker but still catches
        source edits.
        """
        if adapter_name == "rust":
            return self._hash_rust_crate(project)
        elif adapter_name == "typescript":
            return self._hash_typescript_package(project)
        elif adapter_name == "python":
            return self._hash_python_package(project)
        else:
            return self._hash_generic_adapter(project, adapter_name)

    def _hash_sources(self, hasher, root: Path, extensions: tuple[str, ...]) -> None:
        """Add to ``hasher`` each file under ``root`` with one of
        ``extensions``, outside hidden and excluded directories, in sorted
        order: its path relative to ``root``, so renames are detected, then
        its bytes."""
        for file in sorted(f for ext in extensions for f in root.rglob(f"*{ext}")):
            relative = file.relative_to(root)
            if not file.is_file() or any(
                part.startswith(".") or part in self.excluded_dirs
                for part in relative.parts[:-1]
            ):
                continue
            try:
                hasher.update(str(relative).encode())
                hasher.update(file.read_bytes())
            except OSError:
                pass

    def _hash_generic_adapter(self, project: str, adapter_name: str) -> str:
        """Fallback hasher for adapters without bespoke logic.

        Hashes every source file (by the adapter's declared extensions)
        under ``project_root/project`` — or the root itself when project
        equals the root basename. Dep-manifest changes without a source
        edit won't invalidate the cache; accepted trade-off until each
        adapter gets its own hasher.
        """
        candidate = self.project_root / project
        pkg_path = candidate if candidate.is_dir() else self.project_root

        hasher = hashlib.sha256()
        hasher.update(f"adapter:{adapter_name}".encode())
        self._hash_sources(hasher, pkg_path, ADAPTERS[adapter_name].extensions)
        return hasher.hexdigest()[:16]

    def _hash_rust_crate(self, crate: str) -> str:
        """Hash Rust crate sources + toolchain for change detection.

        Hashes:
        - Cargo.toml (dependencies affect types)
        - Workspace Cargo.lock (if present) — resolved dep versions affect types
        - All .rs files in the crate, outside hidden and excluded directories
        - Pinned rust toolchain and scip extra args (so cache invalidates when
          the user changes either)
        """
        crate_path = rust_crate_dir(self.project_root, crate)
        hasher = hashlib.sha256()

        # Hash pinned toolchain + extra args (cache-key component).
        hasher.update(f"toolchain:{self._scip_toolchain or ''}".encode())
        hasher.update(f"extra_args:{'|'.join(self._scip_extra_args)}".encode())

        # Hash Cargo.toml
        cargo_toml = crate_path / "Cargo.toml"
        if cargo_toml.is_file():
            hasher.update(cargo_toml.read_bytes())

        # Hash workspace Cargo.lock (resolved dependency versions affect types).
        cargo_lock = self.project_root / "Cargo.lock"
        if cargo_lock.is_file():
            hasher.update(cargo_lock.read_bytes())

        self._hash_sources(hasher, crate_path, (".rs",))
        return hasher.hexdigest()[:16]

    def _hash_typescript_package(self, package: str) -> str:
        """Hash TypeScript/JavaScript package sources for change detection.

        Hashes:
        - package.json (dependencies affect types)
        - tsconfig.json if present
        - All .ts, .tsx, .js, .jsx files under the package root

        Args:
            package: Name of the package to hash

        Returns:
            16-character hex hash string
        """
        pkg_path = project_dir(self.project_root, package, "package.json")
        hasher = hashlib.sha256()

        # Hash package.json
        package_json = pkg_path / "package.json"
        if package_json.is_file():
            hasher.update(package_json.read_bytes())

        # Hash tsconfig.json if present
        tsconfig = pkg_path / "tsconfig.json"
        if tsconfig.is_file():
            hasher.update(tsconfig.read_bytes())

        # The indexer reads the package, not just src/ (commander's lib/, a
        # root index.ts).
        self._hash_sources(hasher, pkg_path, (".ts", ".tsx", ".js", ".jsx"))
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
            if marker.is_file():
                hasher.update(f"{name}:".encode())
                hasher.update(marker.read_bytes())

        self._hash_sources(hasher, pkg_path, (".py",))
        return hasher.hexdigest()[:16]

    def project_dirs(self, scip_paths: dict[str, Path]) -> dict[Path, str]:
        """Map each Rust index of ``update_all()``'s result to its crate's
        directory relative to the project root, which rust-analyzer's
        document paths are relative to."""
        dirs = {}
        for key, scip_file in scip_paths.items():
            adapter_name, _, crate = key.partition(":")
            if adapter_name == "rust":
                crate_path = rust_crate_dir(self.project_root, crate)
                relative = crate_path.relative_to(self.project_root).as_posix()
                dirs[scip_file] = "" if relative == "." else relative
        return dirs

    def _get_scip_paths(self, adapter_name: str, crates: list[str]) -> dict[str, Path]:
        """Get paths to existing SCIP files.

        Args:
            adapter_name: Adapter the crates belong to
            crates: List of crate names

        Returns:
            Dictionary mapping ``<adapter>:<crate>`` keys to SCIP file paths
            (only existing files)
        """
        paths = {}
        for crate in crates:
            scip_file = self._index_path(adapter_name, crate)
            if scip_file.exists():
                paths[self._project_key(adapter_name, crate)] = scip_file
        return paths

    def _load_checksums(self) -> dict[str, str]:
        """Load cached checksums from disk."""
        try:
            with open_regular(self.checksums_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (ValueError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_checksums(self, checksums: dict[str, str]):
        """Save checksums to disk atomically (tmp then rename)."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = create_sibling(self.checksums_file)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(checksums, f, indent=2)
            os.replace(tmp, self.checksums_file)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
