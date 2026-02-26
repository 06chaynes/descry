"""Per-project SCIP generation with incremental caching.

This module manages SCIP index generation for multiple languages with smart caching
to avoid regenerating indices for unchanged projects.

Supported:
- Rust crates (via rust-analyzer)
- TypeScript/JavaScript packages (via scip-typescript)

Performance optimizations:
- Pre-warm rust-analyzer cache with parallel workers
- Run Rust and TypeScript generation concurrently
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

    _DEFAULT_EXCLUDED_DIRS = {"target", "node_modules", "dist", "docs", ".git", "__pycache__", "build", "vendor"}

    def __init__(self, project_root: Path, excluded_dirs: set[str] | None = None, scip_timeout_minutes: int | None = None):
        """Initialize the cache manager.

        Args:
            project_root: Root directory of the project
            excluded_dirs: Directory names to skip during discovery.
                Defaults to a standard set including target, node_modules,
                dist, docs, .git, __pycache__, build, and vendor.
            scip_timeout_minutes: Timeout in minutes for SCIP generation.
                0 means unlimited. None means use env var or default.
        """
        self.project_root = project_root
        self.cache_dir = project_root / ".descry_cache" / "scip"
        self.checksums_file = self.cache_dir / "checksums.json"
        self.excluded_dirs = excluded_dirs if excluded_dirs is not None else self._DEFAULT_EXCLUDED_DIRS
        self._scip_timeout_minutes = scip_timeout_minutes

    def get_projects(self) -> List[Tuple[str, str]]:
        """Auto-discover all indexable projects.

        Returns:
            List of (project_name, project_type) tuples.
            project_type is one of: "rust", "typescript"
        """
        projects = []
        projects.extend((name, "rust") for name in self.get_rust_crates())
        projects.extend((name, "typescript") for name in self.get_typescript_packages())
        return sorted(projects)

    def get_crates(self) -> List[str]:
        """Auto-discover Rust crates (backwards compatibility alias)."""
        return self.get_rust_crates()

    def get_rust_crates(self) -> List[str]:
        """Auto-discover Rust crates in workspace.

        Finds directories containing Cargo.toml with a src/ subdirectory.

        Returns:
            List of crate directory names (e.g., ['backend', 'database', 'api'])
        """
        crates = []

        # Check for workspace members in root Cargo.toml
        root_cargo = self.project_root / "Cargo.toml"
        if root_cargo.exists():
            for cargo_toml in self.project_root.glob("*/Cargo.toml"):
                crate_dir = cargo_toml.parent
                # Skip hidden directories and common non-crate directories
                if crate_dir.name.startswith("."):
                    continue
                if crate_dir.name in self.excluded_dirs:
                    continue
                if (crate_dir / "src").exists():
                    crates.append(crate_dir.name)

        return sorted(crates)

    def get_typescript_packages(self) -> List[str]:
        """Auto-discover TypeScript/JavaScript packages.

        Finds directories containing package.json with TypeScript/JavaScript source files.

        Returns:
            List of package directory names (e.g., ['webapp', 'dashboard'])
        """
        packages = []

        for package_json in self.project_root.glob("*/package.json"):
            pkg_dir = package_json.parent
            # Skip hidden directories
            if pkg_dir.name.startswith("."):
                continue
            # Skip excluded directories
            if pkg_dir.name in self.excluded_dirs:
                continue
            # Check for TypeScript/JavaScript source files
            has_ts = list(pkg_dir.glob("src/**/*.ts")) or list(pkg_dir.glob("src/**/*.tsx"))
            has_js = list(pkg_dir.glob("src/**/*.js")) or list(pkg_dir.glob("src/**/*.jsx"))
            if has_ts or has_js:
                packages.append(pkg_dir.name)

        return sorted(packages)

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

    def update_changed(self, parallel: bool = True) -> Dict[str, Path]:
        """Update SCIP for changed Rust crates only (backwards compatible).

        Args:
            parallel: Whether to generate SCIP for multiple crates in parallel

        Returns:
            Dictionary mapping crate names to their SCIP file paths
        """
        return self.update_changed_rust(parallel)

    def update_changed_rust(self, parallel: bool = True) -> Dict[str, Path]:
        """Update SCIP for changed Rust crates only.

        Pre-warms rust-analyzer cache before generation for better performance.

        Args:
            parallel: Whether to generate SCIP for multiple crates in parallel

        Returns:
            Dictionary mapping crate names to their SCIP file paths
        """
        crates = self.get_rust_crates()
        if not crates:
            logger.info("No Rust crates found in project")
            return {}

        changed = [c for c in crates if self.needs_update(c, "rust")]

        if changed:
            logger.info(f"SCIP: Regenerating for {len(changed)} changed Rust crate(s): {changed}")
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Pre-warm cache if generating multiple crates (amortizes the cost)
            if len(changed) > 1:
                self._prime_rust_analyzer_cache()

            if parallel and len(changed) > 1:
                max_workers = self._get_max_workers(len(changed))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(self._generate_rust_scip, changed))
            else:
                results = [self._generate_rust_scip(c) for c in changed]

            # Update checksums for successfully generated crates
            checksums = self._load_checksums()
            for crate, success in zip(changed, results):
                if success:
                    checksums[crate] = self._hash_project(crate, "rust")
            self._save_checksums(checksums)
        else:
            logger.debug("SCIP: All Rust crates up-to-date")

        return self._get_scip_paths(crates)

    def update_changed_typescript(self, parallel: bool = False) -> Dict[str, Path]:
        """Update SCIP for changed TypeScript packages.

        Args:
            parallel: Whether to generate SCIP for multiple packages in parallel

        Returns:
            Dictionary mapping package names to their SCIP file paths
        """
        packages = self.get_typescript_packages()
        if not packages:
            logger.info("No TypeScript packages found in project")
            return {}

        changed = [p for p in packages if self.needs_update(p, "typescript")]

        if changed:
            logger.info(f"SCIP: Regenerating for {len(changed)} changed TypeScript package(s): {changed}")
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            if parallel and len(changed) > 1:
                max_workers = self._get_max_workers(len(changed))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(self._generate_typescript_scip, changed))
            else:
                results = [self._generate_typescript_scip(p) for p in changed]

            # Update checksums for successfully generated packages
            checksums = self._load_checksums()
            for pkg, success in zip(changed, results):
                if success:
                    checksums[pkg] = self._hash_project(pkg, "typescript")
            self._save_checksums(checksums)
        else:
            logger.debug("SCIP: All TypeScript packages up-to-date")

        return self._get_scip_paths(packages)

    def update_all(self, parallel: bool = False) -> Dict[str, Path]:
        """Update SCIP for all changed projects (Rust and TypeScript).

        Runs Rust and TypeScript generation concurrently since they
        use different tools with no shared resources.

        Args:
            parallel: Whether to generate SCIP in parallel within each language

        Returns:
            Dictionary mapping project names to their SCIP file paths
        """
        results = {}

        # Run Rust and TypeScript generation concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(self.update_changed_rust, parallel): "rust",
                executor.submit(self.update_changed_typescript, parallel): "typescript",
            }

            for future in as_completed(futures):
                lang = futures[future]
                try:
                    lang_results = future.result()
                    results.update(lang_results)
                except Exception as e:
                    logger.error(f"SCIP: {lang} generation failed: {e}")

        return results

    def _generate_rust_scip(self, crate: str) -> bool:
        """Generate SCIP for a single Rust crate.

        Uses rust-analyzer's scip command to generate the index.

        Args:
            crate: Name of the crate to generate SCIP for

        Returns:
            True if generation succeeded, False otherwise
        """
        crate_path = self.project_root / crate
        output_path = self.cache_dir / f"{crate}.scip"

        # First-time generation is much slower as rust-analyzer builds workspace analysis
        # Check if any SCIP files exist to determine if this is likely a first run
        existing_scip = list(self.cache_dir.glob("*.scip")) if self.cache_dir.exists() else []
        is_first_run = len(existing_scip) == 0

        timeout_seconds = self._get_timeout(is_first_run)
        timeout_str = "unlimited" if timeout_seconds is None else f"{timeout_seconds//60}min"
        logger.info(f"SCIP: Generating index for {crate}... (timeout: {timeout_str})")

        try:
            # rust-analyzer scip runs from workspace root to share analysis cache
            result = subprocess.run(
                [
                    "rust-analyzer",
                    "scip",
                    str(crate_path),
                    "--output",
                    str(output_path),
                    "--exclude-vendored-libraries",  # Skip vendored code for speed
                ],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=str(self.project_root),
            )

            if result.returncode == 0 and output_path.exists():
                size_kb = output_path.stat().st_size / 1024
                logger.info(f"SCIP: Generated {crate}.scip ({size_kb:.1f} KB)")
                return True
            else:
                logger.warning(
                    f"SCIP: Failed to generate for {crate}: {result.stderr[:200]}"
                )
                return False

        except subprocess.TimeoutExpired:
            logger.warning(f"SCIP: Generation timed out for {crate}")
            return False
        except Exception as e:
            logger.warning(f"SCIP: Error generating for {crate}: {e}")
            return False

    def _generate_typescript_scip(self, package: str) -> bool:
        """Generate SCIP for a single TypeScript/JavaScript package.

        Uses scip-typescript to generate the index.

        Args:
            package: Name of the package to generate SCIP for

        Returns:
            True if generation succeeded, False otherwise
        """
        package_path = self.project_root / package
        output_path = self.cache_dir / f"{package}.scip"

        # Check if this is a first run for timeout calculation
        existing_scip = list(self.cache_dir.glob("*.scip")) if self.cache_dir.exists() else []
        is_first_run = len(existing_scip) == 0

        timeout_seconds = self._get_timeout(is_first_run)
        timeout_str = "unlimited" if timeout_seconds is None else f"{timeout_seconds//60}min"
        logger.info(f"SCIP: Generating TypeScript index for {package}... (timeout: {timeout_str})")

        # Build command with optional flags
        cmd = [
            "scip-typescript",
            "index",
            "--output",
            str(output_path),
        ]

        # Add --infer-tsconfig for SvelteKit projects that may not have
        # a standard tsconfig.json at root, or have complex extends chains
        svelte_config = package_path / "svelte.config.js"
        vite_config = package_path / "vite.config.ts"
        if svelte_config.exists() or vite_config.exists():
            cmd.append("--infer-tsconfig")
            logger.debug(f"SCIP: Using --infer-tsconfig for SvelteKit project {package}")

        try:
            # scip-typescript needs to be run from the package directory
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=str(package_path),
            )

            if result.returncode == 0 and output_path.exists():
                size_kb = output_path.stat().st_size / 1024
                logger.info(f"SCIP: Generated {package}.scip ({size_kb:.1f} KB)")
                return True
            else:
                logger.warning(
                    f"SCIP: Failed to generate for {package}: {result.stderr[:200]}"
                )
                return False

        except subprocess.TimeoutExpired:
            logger.warning(f"SCIP: Generation timed out for {package}")
            return False
        except Exception as e:
            logger.warning(f"SCIP: Error generating for {package}: {e}")
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
                env={**os.environ, "RUST_ANALYZER_THREADS": str(num_threads)},
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

    def _get_timeout(self, is_first_run: bool = False) -> int | None:
        """Get timeout in seconds based on config, environment, and run type.

        Priority: config value > env var > default (no timeout)
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
        else:
            return int(env_timeout) * 60

    def _hash_project(self, project: str, project_type: str = "rust") -> str:
        """Hash project sources for change detection.

        Args:
            project: Name of the project to hash
            project_type: Type of project ("rust" or "typescript")

        Returns:
            16-character hex hash string
        """
        if project_type == "rust":
            return self._hash_rust_crate(project)
        elif project_type == "typescript":
            return self._hash_typescript_package(project)
        else:
            raise ValueError(f"Unknown project type: {project_type}")

    def _hash_rust_crate(self, crate: str) -> str:
        """Hash Rust crate sources for change detection.

        Hashes:
        - Cargo.toml (dependencies affect types)
        - All .rs files in the crate

        Args:
            crate: Name of the crate to hash

        Returns:
            16-character hex hash string
        """
        crate_path = self.project_root / crate
        hasher = hashlib.sha256()

        # Hash Cargo.toml
        cargo_toml = crate_path / "Cargo.toml"
        if cargo_toml.exists():
            hasher.update(cargo_toml.read_bytes())

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
                with open(self.checksums_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_checksums(self, checksums: Dict[str, str]):
        """Save checksums to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.checksums_file, "w") as f:
            json.dump(checksums, f, indent=2)

    def get_all_scip_files(self) -> List[Path]:
        """Get all existing SCIP files in the cache.

        Returns:
            List of paths to SCIP files
        """
        if not self.cache_dir.exists():
            return []
        return sorted(self.cache_dir.glob("*.scip"))

    def clear_cache(self):
        """Clear all cached SCIP files and checksums."""
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            logger.info("SCIP: Cache cleared")

    def validate_coverage(self) -> Dict[str, dict]:
        """Validate SCIP coverage for all projects.

        Checks each SCIP file exists and reports basic statistics.

        Returns:
            Dictionary mapping project names to coverage info:
            {
                "project_name": {
                    "exists": bool,
                    "size_kb": float,
                    "project_type": str,
                }
            }
        """
        coverage = {}

        for project, project_type in self.get_projects():
            scip_file = self.cache_dir / f"{project}.scip"
            if scip_file.exists():
                size_kb = scip_file.stat().st_size / 1024
                coverage[project] = {
                    "exists": True,
                    "size_kb": round(size_kb, 1),
                    "project_type": project_type,
                }
            else:
                coverage[project] = {
                    "exists": False,
                    "size_kb": 0,
                    "project_type": project_type,
                }

        return coverage
