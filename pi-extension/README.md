# Descry Pi Extension

Pi extension that provides codebase knowledge graph tools via `descry_*` tool registrations.

## Prerequisites

- Python 3.11+
- descry package installed

## Install

1. Install the descry package:

```bash
cd ~/Documents/descry
pip install -e ".[all]"
```

2. Symlink the extension into your Pi extensions directory:

```bash
ln -s ~/Documents/descry/pi-extension ~/.pi/extensions/descry-tools
```

Or copy the files:

```bash
cp ~/Documents/descry/pi-extension/descry-tools.ts ~/.pi/extensions/
cp ~/Documents/descry/pi-extension/descry-cli.py ~/.pi/extensions/
```

3. Restart Pi and run `/descry-setup` to verify.

## Tools

All tools are prefixed with `descry_`:

- `descry_ensure` - Ensure graph exists and is fresh
- `descry_search` - Search symbols (keyword + semantic)
- `descry_callers` - Find callers of a symbol
- `descry_callees` - Find what a symbol calls
- `descry_context` - Get full context for a symbol
- `descry_structure` - Show file structure overview
- `descry_quick` - Find + context in one step
- `descry_health` - Health check
- `descry_index` - Regenerate graph
- `descry_impls` - Find trait implementations
- `descry_cross_lang` - Trace frontend/backend API calls
- `descry_churn` - Find code churn hotspots
- `descry_evolution` - Track symbol changes over time
- `descry_changes` - Analyze change impact
- `descry_flow` - Trace call flow
- `descry_path` - Find shortest call path
- `descry_flatten` - Show class API with inherited methods
- `descry_semantic` - Pure semantic search
