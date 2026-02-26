"""Tests for descry.mcp_server — MCP server registration."""


def test_mcp_server_importable():
    from descry.mcp_server import mcp

    assert mcp is not None
    assert mcp.name == "descry"
