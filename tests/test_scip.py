"""Tests for descry.scip — SCIP parsing, import resolution, and cache performance."""

import multiprocessing
import os
from pathlib import Path


from descry.scip.parser import ScipIndex
from descry.scip.cache import ScipCacheManager
from descry.generate import TypeScriptSymbolTable


class TestTypescriptScipParsing:
    """Tests for TypeScript SCIP symbol parsing."""

    def test_simple_function(self):
        """Should extract simple function name from TypeScript SCIP symbol."""
        # Create a minimal index to test parsing
        index = ScipIndex([])

        # Test the internal descriptor parsing
        descriptors = "src/lib/api/`client.ts`/getAuthToken()."
        result = index._parse_typescript_descriptors(descriptors)
        assert result == ["getAuthToken"], f"Expected ['getAuthToken'], got {result}"

    def test_class_method(self):
        """Should extract class and method from TypeScript SCIP symbol."""
        index = ScipIndex([])

        descriptors = "src/lib/stores/`users.ts`/UsersStore#fetchUsers()."
        result = index._parse_typescript_descriptors(descriptors)
        assert result == ["UsersStore", "fetchUsers"], (
            f"Expected ['UsersStore', 'fetchUsers'], got {result}"
        )

    def test_multiple_backticks(self):
        """Should handle multiple backtick segments in path."""
        index = ScipIndex([])

        descriptors = "src/lib/`stores`/`auth.ts`/AuthStore#login()."
        result = index._parse_typescript_descriptors(descriptors)
        assert result == ["AuthStore", "login"], (
            f"Expected ['AuthStore', 'login'], got {result}"
        )

    def test_nested_class_method(self):
        """Should handle nested type/method symbols."""
        index = ScipIndex([])

        descriptors = "src/`client.ts`/ApiClient#request()."
        result = index._parse_typescript_descriptors(descriptors)
        assert "ApiClient" in result
        assert "request" in result

    def test_export_function(self):
        """Should handle exported function symbols."""
        index = ScipIndex([])

        descriptors = "src/lib/`utils.ts`/formatDate()."
        result = index._parse_typescript_descriptors(descriptors)
        assert result == ["formatDate"], f"Expected ['formatDate'], got {result}"


class TestTypescriptImportResolution:
    """Tests for TypeScript import resolution."""

    def test_symbol_table_basic(self):
        """Should track basic imports."""
        table = TypeScriptSymbolTable(
            "/project/lens/src/lib/stores/test.ts", "/project"
        )
        table.imports = {
            "fetchData": ("./api", "named"),
            "apiClient": ("$lib/api/client", "default"),
        }

        # Should be able to get import source
        source = table.get_import_source("fetchData")
        assert source == "./api"

        source = table.get_import_source("apiClient")
        assert source == "$lib/api/client"

    def test_namespace_import(self):
        """Should identify namespace imports."""
        table = TypeScriptSymbolTable(
            "/project/lens/src/lib/stores/test.ts", "/project"
        )
        table.namespaces = {"schedulesApi": "$lib/api/schedules"}
        table.imports = {"schedulesApi": ("$lib/api/schedules", "namespace")}

        # schedulesApi.list() should be identified as a namespace call
        assert table.is_namespace_call("schedulesApi.list") is True
        assert table.is_namespace_call("regularFunc") is False

        # Should get the import source
        source = table.get_import_source("schedulesApi")
        assert source == "$lib/api/schedules"

    def test_type_import_skipped(self):
        """Should identify type-only imports."""
        table = TypeScriptSymbolTable("/project/lens/src/test.ts", "/project")
        table.imports = {
            "MyType": ("./types", "type"),
            "myFunc": ("./utils", "named"),
        }

        assert table.is_type_import("MyType") is True
        assert table.is_type_import("myFunc") is False

    def test_unknown_import(self):
        """Should return None for unknown imports."""
        table = TypeScriptSymbolTable("/project/lens/src/test.ts", "/project")
        table.imports = {}

        assert table.get_import_source("unknownFunc") is None
        assert table.is_type_import("unknownFunc") is False


class TestScipCachePerformance:
    """Tests for SCIP cache performance optimizations."""

    def test_get_max_workers_env_override(self):
        """Should respect DESCRY_SCIP_WORKERS env variable."""
        # Save original
        original = os.environ.get("DESCRY_SCIP_WORKERS")

        try:
            os.environ["DESCRY_SCIP_WORKERS"] = "1"
            manager = ScipCacheManager(Path("."))
            assert manager._get_max_workers(10) == 1

            os.environ["DESCRY_SCIP_WORKERS"] = "5"
            assert manager._get_max_workers(10) == 5
            # Should cap at num_items
            assert manager._get_max_workers(3) == 3
        finally:
            if original is None:
                os.environ.pop("DESCRY_SCIP_WORKERS", None)
            else:
                os.environ["DESCRY_SCIP_WORKERS"] = original

    def test_get_max_workers_defaults_to_reasonable_value(self):
        """Should default to reasonable worker count without env override."""
        # Ensure env var is not set
        original = os.environ.pop("DESCRY_SCIP_WORKERS", None)
        try:
            manager = ScipCacheManager(Path("."))
            workers = manager._get_max_workers(10)
            # Should return between 2 and 4 (reasonable defaults)
            assert 2 <= workers <= 4
        finally:
            if original is not None:
                os.environ["DESCRY_SCIP_WORKERS"] = original

    def test_get_prime_threads_env_override(self):
        """Should respect DESCRY_PRIME_THREADS env variable."""
        original = os.environ.get("DESCRY_PRIME_THREADS")
        try:
            os.environ["DESCRY_PRIME_THREADS"] = "8"
            manager = ScipCacheManager(Path("."))
            assert manager._get_prime_threads() == 8
        finally:
            if original is None:
                os.environ.pop("DESCRY_PRIME_THREADS", None)
            else:
                os.environ["DESCRY_PRIME_THREADS"] = original

    def test_get_prime_threads_defaults_based_on_cpu(self):
        """Should default based on CPU count."""
        original = os.environ.pop("DESCRY_PRIME_THREADS", None)
        try:
            manager = ScipCacheManager(Path("."))
            threads = manager._get_prime_threads()
            expected = max(2, multiprocessing.cpu_count() - 2)
            assert threads == expected
        finally:
            if original is not None:
                os.environ["DESCRY_PRIME_THREADS"] = original
