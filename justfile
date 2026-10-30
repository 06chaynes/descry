# Descry — codebase knowledge graph toolkit

# Install in editable mode with all dependencies
install:
    uv venv .venv
    . .venv/bin/activate && uv pip install -e ".[dev]"

# Run tests: all, one file, one test, or by expression
[positional-arguments]
test *args:
    . .venv/bin/activate && if [ "$#" -eq 0 ]; then set -- tests/; fi && python -m pytest "$@"

# Run tests with verbose output
test-v:
    . .venv/bin/activate && python -m pytest tests/ -v

# Index the current project
[positional-arguments]
index *args:
    . .venv/bin/activate && descry index "$@"

# Health check
health:
    . .venv/bin/activate && descry health

# Search symbols
[positional-arguments]
search query *args:
    . .venv/bin/activate && descry search "$@"

# Launch web UI (default: http://127.0.0.1:8787)
[positional-arguments]
web *args:
    . .venv/bin/activate && descry-web "$@"

# Start MCP server
mcp:
    . .venv/bin/activate && descry-mcp

# Lint with ruff
lint:
    . .venv/bin/activate && ruff check src/ tests/ tools/

# Format with ruff
fmt:
    . .venv/bin/activate && ruff format src/ tests/ tools/

# Clean build artifacts and caches
clean:
    rm -rf .descry_cache .pytest_cache dist build src/*.egg-info
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
