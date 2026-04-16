# Descry — codebase knowledge graph toolkit

# Install in editable mode with all dependencies
install:
    uv venv .venv
    . .venv/bin/activate && uv pip install -e ".[dev]"

# Run tests
test *args:
    . .venv/bin/activate && python -m pytest tests/ {{args}}

# Run tests with verbose output
test-v:
    . .venv/bin/activate && python -m pytest tests/ -v

# Index the current project
index *args:
    . .venv/bin/activate && descry index {{args}}

# Health check
health:
    . .venv/bin/activate && descry health

# Search symbols
search query *args:
    . .venv/bin/activate && descry search "{{query}}" {{args}}

# Launch web UI (default: http://127.0.0.1:8787)
web *args:
    . .venv/bin/activate && descry-web {{args}}

# Start MCP server
mcp:
    . .venv/bin/activate && descry-mcp

# Lint with ruff
lint:
    . .venv/bin/activate && ruff check src/ tests/

# Format with ruff
fmt:
    . .venv/bin/activate && ruff format src/ tests/

# Clean build artifacts and caches
clean:
    rm -rf .descry_cache .pytest_cache dist build src/*.egg-info
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
