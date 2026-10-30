"""The tool reference is what its generator writes from the code now."""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not importlib.util.find_spec("mcp"), reason="needs mcp")
def test_the_tool_reference_is_current(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "gen_tools", REPO / "tools" / "docs" / "gen_tools.py"
    )
    gen_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_tools)
    gen_tools.write(tmp_path / "TOOLS.md")
    written = (tmp_path / "TOOLS.md").read_text()
    assert written == (REPO / "docs" / "TOOLS.md").read_text(), (
        "docs/TOOLS.md is stale: run tools/docs/gen_tools.py docs/TOOLS.md"
    )
