# Descry Encyclopedia Panel — Design

## Overview

Replace the current minimal help modal (symbol legend + 3 tips) with a full-panel encyclopedia overlay inspired by the Stationpedia from Stationeers. Provides searchable, categorized reference for every tool, key concepts, and multi-tool workflows.

## Requirements

- **Full panel overlay** — slides over the main UI, dismissable
- **Context-aware** — opens to the article for the currently active tool
- **Searchable** — real-time filtering of all articles by title/keywords
- **Three content categories:** Tools, Concepts, Workflows
- **Reference-card depth** — ~1 screen per article (not tutorial-length)
- **Pure Alpine.js** — all content inline in `index.html`, no extra dependencies

## Architecture

### Alpine State

```javascript
// Added to the main x-data object:
helpSearch: '',          // Search filter text
helpArticle: null,       // Current article slug (null = TOC view)

// Article index (flat array for search)
helpArticles: [
    { slug: 'search', title: 'Search', category: 'tools', keywords: ['keyword', 'filter', 'regex', 'find'] },
    { slug: 'semantic', title: 'Semantic Search', category: 'tools', keywords: ['vector', 'embedding', 'similar', 'meaning'] },
    // ... all articles
],
```

### Context-Aware Opening

```javascript
openHelp() {
    this.helpSearch = '';
    this.helpArticle = this.activeTool;  // Jump to active tool's article
    this.showHelp = true;
},
openHelpTOC() {
    this.helpSearch = '';
    this.helpArticle = null;  // Show table of contents
    this.showHelp = true;
},
```

### Panel Layout

```
┌─────────────────────────────────────────┐
│ [← TOC]  Descry Encyclopedia  [____] [×]│
├─────────────────────────────────────────┤
│                                         │
│  When helpArticle === null (TOC view):  │
│                                         │
│  TOOLS                                  │
│  ├ Search .................. keyword    │
│  ├ Semantic Search ......... vector     │
│  ├ Quick Lookup ............ symbol     │
│  ├ Callers ................. who calls  │
│  ├ Callees ................. what calls │
│  ├ Context ................. full node  │
│  ├ Structure ............... file map   │
│  ├ Flatten ................. hierarchy  │
│  ├ Implementations ......... traits     │
│  ├ Flow .................... dataflow   │
│  ├ Path .................... reachable  │
│  ├ Cross-Language .......... API trace  │
│  ├ Churn ................... hotspots   │
│  ├ Evolution ............... timeline   │
│  ├ Changes ................. impact     │
│  ├ Health .................. status     │
│  ├ Reindex ................. rebuild    │
│  └ Source .................. view code  │
│                                         │
│  CONCEPTS                               │
│  ├ The Knowledge Graph ..... nodes/edges│
│  ├ Symbol Types ............ FUN/MET/.. │
│  ├ SCIP Indexing ........... precision  │
│  ├ Semantic Search ......... embeddings │
│  └ Git History Analysis .... churn      │
│                                         │
│  WORKFLOWS                              │
│  ├ Investigating a Function             │
│  ├ Understanding a Codebase             │
│  └ Tracking Recent Changes              │
│                                         │
│  When helpArticle !== null:             │
│  Shows the selected article content     │
│                                         │
└─────────────────────────────────────────┘
```

### Search Behavior

The search input filters the TOC entries in real-time. Matches against `title` and `keywords`. When searching, articles are shown in a flat list (no category headers for non-matching entries). Clicking a result navigates to that article.

## Content Catalog

### Tools (18 articles)

Each tool article follows this template:

```
## [Title]
[One-line description]

### Parameters
| Field | Description |
|-------|-------------|
| name  | Symbol name to search for |
| limit | Max results (default: 20) |

### Example
Search for "parse" → returns functions, methods, and classes matching "parse"

### Tips
- Use this when you know the exact symbol name
- For fuzzy/conceptual searches, try Semantic Search instead
```

Tool list:
1. **Search** — keyword search across all symbols
2. **Semantic Search** — vector similarity search by meaning
3. **Quick Lookup** — fast symbol lookup by exact name
4. **Callers** — find all callers of a function/method
5. **Callees** — find all functions called by a symbol
6. **Context** — full node details (signature, docstring, source, edges)
7. **Structure** — list all symbols defined in a file
8. **Flatten** — inheritance/trait hierarchy for a type
9. **Implementations** — find all implementations of a trait/interface method
10. **Flow** — forward/backward dataflow from a symbol
11. **Path** — find call paths between two symbols
12. **Cross-Language** — trace API calls across frontend/backend boundaries
13. **Churn** — identify frequently changed files and symbols
14. **Evolution** — timeline of changes to a specific file or symbol
15. **Changes** — impact analysis of recent commits
16. **Health** — graph status (node/edge counts, staleness)
17. **Reindex** — rebuild the knowledge graph
18. **Source** — view source code with syntax context

### Concepts (5 articles)

1. **The Knowledge Graph** — what nodes and edges represent, how the graph is built from source code
2. **Symbol Types** — FUN, MET, CLA, CON, FIL, CFG explained (migrates current legend)
3. **SCIP Indexing** — what SCIP is, why it provides precise cross-references, which languages are supported
4. **Semantic Embeddings** — how vector search works, when to use it vs keyword search
5. **Git History Analysis** — how descry uses git log for churn, evolution, and change impact

### Workflows (3 articles)

1. **Investigating a Function** — Quick Lookup → Context → Callers → Flow. "Start with the name, expand outward."
2. **Understanding a Codebase** — Health → Structure (key files) → Search (entry points) → Flow. "Get the lay of the land."
3. **Tracking Recent Changes** — Changes (HEAD~5..HEAD) → Churn → Evolution. "See what's been happening."

## Styling

### Panel

- Full-viewport overlay with semi-transparent backdrop (reuses existing `help-modal-overlay` pattern)
- Panel: `max-width: 700px`, centered, `max-height: 85vh`, scrollable body
- Header: sticky, contains back button, title, search input, close button
- Respects existing dark theme

### TOC Entries

- Clickable rows with title + brief description
- Category headers (`TOOLS`, `CONCEPTS`, `WORKFLOWS`) as section dividers
- Hover highlight matching existing sidebar style

### Article View

- Title as `<h2>`
- Parameters as a compact table
- Example in a highlighted box
- Tips as a bulleted list
- Back-to-TOC button always visible in header

## Implementation Notes

- All article content is defined as HTML blocks with `id="help-{slug}"`, shown/hidden via `x-show="helpArticle === '{slug}'"`
- The TOC is generated by iterating `helpArticles` array, filtered by `helpSearch`
- No new API endpoints needed
- No new CSS files — styles added to existing `<style>` block
- The existing symbol legend and tips are absorbed into the "Symbol Types" concept article
