"""Tests for descry.mcp_server — MCP server tool registration and enums."""

import pytest


class TestMcpServerRegistration:
    def test_server_importable(self):
        from descry.mcp_server import mcp

        assert mcp is not None
        assert mcp.name == "descry"

    def test_all_tools_registered(self):
        from descry.mcp_server import mcp

        tool_names = set()
        for tool in mcp._tool_manager._tools.values():
            tool_names.add(tool.name)

        expected = {
            "descry_health",
            "descry_ensure",
            "descry_status",
            "descry_callers",
            "descry_callees",
            "descry_context",
            "descry_flow",
            "descry_search",
            "descry_structure",
            "descry_flatten",
            "descry_index",
            "descry_semantic",
            "descry_quick",
            "descry_impls",
            "descry_path",
            "descry_cross_lang",
            "descry_churn",
            "descry_evolution",
            "descry_changes",
        }
        assert tool_names == expected, (
            f"Missing: {expected - tool_names}, Extra: {tool_names - expected}"
        )

    def test_tool_count(self):
        from descry.mcp_server import mcp

        tools = list(mcp._tool_manager._tools.values())
        assert len(tools) == 19


class TestMcpEnums:
    def test_direction_values(self):
        from descry.mcp_server import Direction

        assert Direction.forward.value == "forward"
        assert Direction.backward.value == "backward"

    def test_language_values(self):
        from descry.mcp_server import Language

        assert Language.rust.value == "rust"
        assert Language.all.value == "all"

    def test_symbol_type_values(self):
        from descry.mcp_server import SymbolType

        assert SymbolType.function.value == "function"
        assert SymbolType.all.value == "all"

    def test_churn_mode_values(self):
        from descry.mcp_server import ChurnMode

        assert ChurnMode.symbols.value == "symbols"
        assert ChurnMode.co_change.value == "co-change"

    def test_cross_lang_mode_values(self):
        from descry.mcp_server import CrossLangMode

        assert CrossLangMode.endpoint.value == "endpoint"
        assert CrossLangMode.list.value == "list"
        assert CrossLangMode.stats.value == "stats"


class TestSvcAssertion:
    def test_svc_raises_when_uninitialized(self):
        from descry.mcp_server import _svc
        import descry.mcp_server as mod

        original = mod._service
        try:
            mod._service = None
            with pytest.raises(RuntimeError, match="Server not initialized"):
                _svc()
        finally:
            mod._service = original
