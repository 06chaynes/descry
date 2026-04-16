# Descry Encyclopedia Panel — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the minimal help modal with a full-panel encyclopedia overlay providing searchable reference for all 18 tools, 5 concept articles, and 3 workflow guides.

**Architecture:** Pure Alpine.js, all content inline in `index.html`. A `helpArticles` data array drives the TOC and search. Article content lives in `<template x-if>` blocks keyed by slug. CSS added to `style.css`. The existing help modal HTML/CSS is replaced entirely.

**Tech Stack:** Alpine.js 3.x (already loaded), HTML, CSS (dark theme variables already defined in `:root`)

**Design doc:** `docs/plans/2026-02-26-help-encyclopedia-design.md`

---

### Task 1: Add encyclopedia CSS to style.css

**Files:**
- Modify: `src/descry/web/web/style.css` (replace lines 120-251, the existing help-btn through help-tip styles)

**Step 1: Replace help modal CSS with encyclopedia panel CSS**

Remove the existing `.help-btn`, `.help-modal-overlay`, `.help-modal`, `.help-modal-header`, `.help-close-btn`, `.help-modal-body`, `.help-legend`, `.help-legend-row`, `.help-legend-name`, `.help-legend-desc`, `.help-divider`, `.help-tips`, `.help-tip` styles (lines 120-251 in `style.css`).

Replace with:

```css
/* --- Help Button --- */
.help-btn {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    border: 1px solid var(--border);
    background: var(--bg-secondary);
    color: var(--text-secondary);
    font-size: 16px;
    font-weight: 700;
    cursor: pointer;
    transition: var(--transition);
    font-family: var(--font-sans);
}
.help-btn:hover {
    color: var(--accent);
    border-color: var(--accent);
}

/* --- Encyclopedia Panel --- */
.ency-overlay {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.6);
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
}
.ency-panel {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    width: 95vw;
    max-width: 700px;
    max-height: 85vh;
    display: flex;
    flex-direction: column;
    box-shadow: 0 16px 48px rgba(0,0,0,0.4);
}

/* Header */
.ency-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
}
.ency-back {
    background: none;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--text-secondary);
    font-size: 13px;
    padding: 4px 10px;
    cursor: pointer;
    transition: var(--transition);
    font-family: var(--font-sans);
    white-space: nowrap;
}
.ency-back:hover { color: var(--accent); border-color: var(--accent); }
.ency-title {
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary);
    white-space: nowrap;
}
.ency-search {
    flex: 1;
    min-width: 0;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--text-primary);
    font-size: 13px;
    padding: 5px 10px;
    font-family: var(--font-sans);
    outline: none;
    transition: var(--transition);
}
.ency-search:focus { border-color: var(--accent); }
.ency-search::placeholder { color: var(--text-muted); }
.ency-close {
    background: none;
    border: none;
    color: var(--text-muted);
    font-size: 22px;
    cursor: pointer;
    padding: 0 4px;
    line-height: 1;
    transition: var(--transition);
}
.ency-close:hover { color: var(--text-primary); }

/* Body (scrollable) */
.ency-body {
    overflow-y: auto;
    padding: 16px 20px 24px;
    flex: 1;
    min-height: 0;
}

/* TOC */
.ency-category {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    padding: 16px 0 6px;
}
.ency-category:first-child { padding-top: 0; }
.ency-toc-item {
    display: flex;
    align-items: baseline;
    gap: 8px;
    padding: 7px 10px;
    border-radius: var(--radius);
    cursor: pointer;
    transition: var(--transition);
}
.ency-toc-item:hover { background: var(--bg-hover); }
.ency-toc-title {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
}
.ency-toc-desc {
    font-size: 12px;
    color: var(--text-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.ency-no-results {
    text-align: center;
    color: var(--text-muted);
    font-size: 13px;
    padding: 40px 0;
}

/* Article */
.ency-article h2 {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 4px;
}
.ency-article .ency-oneliner {
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 16px;
}
.ency-article h3 {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin: 16px 0 6px;
}
.ency-article table {
    width: 100%;
    font-size: 13px;
    border-collapse: collapse;
    margin-bottom: 8px;
}
.ency-article th {
    text-align: left;
    font-weight: 600;
    color: var(--text-secondary);
    padding: 4px 8px;
    border-bottom: 1px solid var(--border);
}
.ency-article td {
    padding: 4px 8px;
    color: var(--text-primary);
    border-bottom: 1px solid var(--bg-tertiary);
}
.ency-article td:first-child {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--accent);
    white-space: nowrap;
}
.ency-example {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 10px 14px;
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 8px;
}
.ency-example code {
    color: var(--accent);
    font-family: var(--font-mono);
    font-size: 12px;
}
.ency-tips {
    list-style: none;
    padding: 0;
}
.ency-tips li {
    font-size: 13px;
    color: var(--text-secondary);
    padding: 3px 0;
    padding-left: 14px;
    position: relative;
}
.ency-tips li::before {
    content: "\2022";
    position: absolute;
    left: 0;
    color: var(--text-muted);
}

/* Symbol legend (concept article) */
.ency-legend-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 5px 0;
    border-bottom: 1px solid var(--bg-tertiary);
}
.ency-legend-row:last-child { border-bottom: none; }
.ency-legend-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
    width: 100px;
}
.ency-legend-desc {
    font-size: 12px;
    color: var(--text-muted);
}

/* Workflow steps */
.ency-steps {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.ency-step {
    display: flex;
    align-items: baseline;
    gap: 8px;
    font-size: 13px;
    color: var(--text-secondary);
}
.ency-step-num {
    font-weight: 700;
    color: var(--accent);
    font-size: 12px;
    flex-shrink: 0;
}
.ency-step code {
    color: var(--accent);
    font-family: var(--font-mono);
    font-size: 12px;
}
```

**Step 2: Verify CSS is valid**

Run: `python3 -c "open('src/descry/web/web/style.css').read(); print('CSS file reads OK')"` from project root.
Expected: `CSS file reads OK`

**Step 3: Commit**

```bash
git add src/descry/web/web/style.css
git commit -m "style: replace help modal CSS with encyclopedia panel styles"
```

---

### Task 2: Add Alpine.js state and helper methods

**Files:**
- Modify: `src/descry/web/web/index.html` (the `app()` function, around lines 1184-1248)

**Step 1: Add encyclopedia state properties**

In the `app()` return object, after the `examples: null,` line (~line 1204), add:

```javascript
// Encyclopedia state
helpSearch: '',
helpArticle: null,
helpArticles: [
    // Tools
    { slug: 'search', title: 'Search', category: 'tools', desc: 'Hybrid keyword + semantic symbol search', keywords: ['keyword', 'filter', 'regex', 'find', 'symbol'] },
    { slug: 'semantic', title: 'Semantic Search', category: 'tools', desc: 'Vector similarity search by meaning', keywords: ['embedding', 'vector', 'natural language', 'meaning', 'conceptual'] },
    { slug: 'quick', title: 'Quick Lookup', category: 'tools', desc: 'Full details on a symbol in one step', keywords: ['symbol', 'lookup', 'details', 'fast'] },
    { slug: 'callers', title: 'Callers', category: 'tools', desc: 'Who calls this function', keywords: ['call', 'reference', 'impact', 'upstream', 'who calls'] },
    { slug: 'callees', title: 'Callees', category: 'tools', desc: 'What this function calls', keywords: ['dependency', 'downstream', 'calls', 'what calls'] },
    { slug: 'context', title: 'Context', category: 'tools', desc: 'Full node dossier: source, callers, callees', keywords: ['details', 'source', 'node', 'dossier'] },
    { slug: 'structure', title: 'Structure', category: 'tools', desc: 'File skeleton with all symbols', keywords: ['file', 'skeleton', 'outline', 'symbols', 'imports'] },
    { slug: 'flatten', title: 'Flatten', category: 'tools', desc: 'Inheritance and trait hierarchy', keywords: ['inheritance', 'trait', 'interface', 'hierarchy', 'class'] },
    { slug: 'impls', title: 'Implementations', category: 'tools', desc: 'Find all implementations of a method', keywords: ['trait', 'interface', 'implement', 'polymorphic', 'concrete'] },
    { slug: 'flow', title: 'Flow Trace', category: 'tools', desc: 'Forward/backward call tree visualization', keywords: ['trace', 'call tree', 'dataflow', 'forward', 'backward'] },
    { slug: 'path', title: 'Call Path', category: 'tools', desc: 'Shortest call chain between two symbols', keywords: ['path', 'route', 'chain', 'between', 'connection'] },
    { slug: 'cross-lang', title: 'Cross-Language', category: 'tools', desc: 'Frontend-to-backend API call tracing', keywords: ['api', 'frontend', 'backend', 'openapi', 'endpoint', 'cross'] },
    { slug: 'churn', title: 'Churn', category: 'tools', desc: 'Most-changed code from git history', keywords: ['hotspot', 'frequency', 'change', 'git', 'coupling'] },
    { slug: 'evolution', title: 'Evolution', category: 'tools', desc: 'Commit timeline for a symbol or file', keywords: ['timeline', 'history', 'commits', 'diff', 'author'] },
    { slug: 'changes', title: 'Changes', category: 'tools', desc: 'Impact analysis of recent commits', keywords: ['impact', 'commit', 'blast radius', 'review', 'callers'] },
    { slug: 'health', title: 'Health', category: 'tools', desc: 'Graph status and feature availability', keywords: ['status', 'graph', 'nodes', 'edges', 'stale'] },
    { slug: 'reindex', title: 'Reindex', category: 'tools', desc: 'Rebuild the knowledge graph', keywords: ['rebuild', 'regenerate', 'refresh', 'index'] },
    { slug: 'source', title: 'Source', category: 'tools', desc: 'View source files with line numbers', keywords: ['file', 'code', 'read', 'view', 'line'] },
    // Concepts
    { slug: 'concept-graph', title: 'The Knowledge Graph', category: 'concepts', desc: 'How nodes and edges model your code', keywords: ['graph', 'node', 'edge', 'model', 'architecture'] },
    { slug: 'concept-symbols', title: 'Symbol Types', category: 'concepts', desc: 'FUN, MET, CLA, CON, FIL, CFG explained', keywords: ['type', 'badge', 'legend', 'function', 'method', 'class'] },
    { slug: 'concept-scip', title: 'SCIP Indexing', category: 'concepts', desc: 'Precise cross-references via compiler data', keywords: ['scip', 'compiler', 'type-aware', 'precise', 'resolution'] },
    { slug: 'concept-embeddings', title: 'Semantic Embeddings', category: 'concepts', desc: 'How vector search finds similar code', keywords: ['vector', 'embedding', 'similarity', 'model', 'cosine'] },
    { slug: 'concept-git', title: 'Git History Analysis', category: 'concepts', desc: 'How descry uses git log for insights', keywords: ['git', 'log', 'blame', 'churn', 'history'] },
    // Workflows
    { slug: 'workflow-investigate', title: 'Investigating a Function', category: 'workflows', desc: 'Start with a name, expand outward', keywords: ['investigate', 'function', 'deep dive', 'understand'] },
    { slug: 'workflow-explore', title: 'Understanding a Codebase', category: 'workflows', desc: 'Get the lay of the land', keywords: ['explore', 'codebase', 'orientation', 'new project'] },
    { slug: 'workflow-changes', title: 'Tracking Recent Changes', category: 'workflows', desc: 'See what has been happening', keywords: ['recent', 'changes', 'review', 'commits', 'impact'] },
],
```

**Step 2: Add helper methods**

After the `ex()` method (~line 1248), add:

```javascript
openHelp() {
    this.helpSearch = '';
    // Context-aware: jump to article for active tool if it exists
    const toolMatch = this.helpArticles.find(a => a.slug === this.activeTool);
    this.helpArticle = toolMatch ? this.activeTool : null;
    this.showHelp = true;
},

openHelpTOC() {
    this.helpSearch = '';
    this.helpArticle = null;
    this.showHelp = true;
},

filteredHelpArticles() {
    if (!this.helpSearch.trim()) return this.helpArticles;
    const q = this.helpSearch.toLowerCase();
    return this.helpArticles.filter(a =>
        a.title.toLowerCase().includes(q) ||
        a.desc.toLowerCase().includes(q) ||
        a.keywords.some(k => k.includes(q))
    );
},
```

**Step 3: Update the help button click handler**

Change the `?` button in the header (line 42) from:
```html
<button class="help-btn" @click="showHelp = !showHelp" title="Symbol legend & tips">?</button>
```
to:
```html
<button class="help-btn" @click="showHelp ? (showHelp = false) : openHelp()" title="Encyclopedia">?</button>
```

**Step 4: Verify the app loads**

Run from project root:
```bash
cd /Users/christian/Documents/descry && python3 -c "
html = open('src/descry/web/web/index.html').read()
assert 'helpArticles' in html, 'helpArticles not found'
assert 'openHelp()' in html, 'openHelp not found'
assert 'filteredHelpArticles()' in html, 'filteredHelpArticles not found'
print('Alpine state additions verified OK')
"
```
Expected: `Alpine state additions verified OK`

**Step 5: Commit**

```bash
git add src/descry/web/web/index.html
git commit -m "feat: add encyclopedia Alpine.js state, article index, and helper methods"
```

---

### Task 3: Build the encyclopedia panel HTML shell (TOC view)

**Files:**
- Modify: `src/descry/web/web/index.html` (replace the existing help modal HTML, lines 1766-1793)

**Step 1: Replace help modal with encyclopedia panel**

Remove the entire `<!-- Help Modal -->` block (lines 1766-1793). Replace with:

```html
<!-- Encyclopedia Panel -->
<div class="ency-overlay"
     x-show="showHelp"
     x-transition.opacity
     @click.self="showHelp = false"
     @keydown.escape.window="showHelp = false"
     style="display:none">
    <div class="ency-panel" @click.stop>
        <!-- Header -->
        <div class="ency-header">
            <button class="ency-back" x-show="helpArticle" @click="helpArticle = null; helpSearch = ''">&#8592; TOC</button>
            <span class="ency-title" x-text="helpArticle ? (helpArticles.find(a => a.slug === helpArticle)?.title || 'Encyclopedia') : 'Descry Encyclopedia'"></span>
            <input class="ency-search" type="text" placeholder="Search articles..."
                   x-model="helpSearch" x-show="!helpArticle"
                   @keydown.escape="helpSearch = ''">
            <button class="ency-close" @click="showHelp = false">&times;</button>
        </div>

        <!-- Body -->
        <div class="ency-body">

            <!-- TOC View -->
            <div x-show="!helpArticle">
                <!-- Grouped by category when not searching -->
                <template x-if="!helpSearch.trim()">
                    <div>
                        <div class="ency-category">Tools</div>
                        <template x-for="a in helpArticles.filter(a => a.category === 'tools')" :key="a.slug">
                            <div class="ency-toc-item" @click="helpArticle = a.slug">
                                <span class="ency-toc-title" x-text="a.title"></span>
                                <span class="ency-toc-desc" x-text="a.desc"></span>
                            </div>
                        </template>
                        <div class="ency-category">Concepts</div>
                        <template x-for="a in helpArticles.filter(a => a.category === 'concepts')" :key="a.slug">
                            <div class="ency-toc-item" @click="helpArticle = a.slug">
                                <span class="ency-toc-title" x-text="a.title"></span>
                                <span class="ency-toc-desc" x-text="a.desc"></span>
                            </div>
                        </template>
                        <div class="ency-category">Workflows</div>
                        <template x-for="a in helpArticles.filter(a => a.category === 'workflows')" :key="a.slug">
                            <div class="ency-toc-item" @click="helpArticle = a.slug">
                                <span class="ency-toc-title" x-text="a.title"></span>
                                <span class="ency-toc-desc" x-text="a.desc"></span>
                            </div>
                        </template>
                    </div>
                </template>

                <!-- Flat filtered list when searching -->
                <template x-if="helpSearch.trim()">
                    <div>
                        <template x-for="a in filteredHelpArticles()" :key="a.slug">
                            <div class="ency-toc-item" @click="helpArticle = a.slug">
                                <span class="ency-toc-title" x-text="a.title"></span>
                                <span class="ency-toc-desc" x-text="a.desc"></span>
                            </div>
                        </template>
                        <div class="ency-no-results" x-show="filteredHelpArticles().length === 0">
                            No articles match your search.
                        </div>
                    </div>
                </template>
            </div>

            <!-- Article Views (shown when helpArticle matches slug) -->
            <!-- PLACEHOLDER: Articles added in Tasks 4-6 -->

        </div>
    </div>
</div>
```

**Step 2: Verify HTML is valid**

Run:
```bash
cd /Users/christian/Documents/descry && python3 -c "
html = open('src/descry/web/web/index.html').read()
assert 'ency-overlay' in html, 'ency-overlay not found'
assert 'ency-toc-item' in html, 'ency-toc-item not found'
assert 'filteredHelpArticles()' in html, 'filteredHelpArticles not found'
# Check old modal is gone
assert 'help-modal-overlay' not in html, 'old help modal still present'
assert 'Symbol Legend' not in html, 'old Symbol Legend still present'
print('Encyclopedia shell verified OK')
"
```
Expected: `Encyclopedia shell verified OK`

**Step 3: Run tests**

Run: `cd /Users/christian/Documents/descry && python3 -m pytest tests/ -x -q`
Expected: All tests pass (the help modal isn't tested by backend tests, so nothing should break)

**Step 4: Commit**

```bash
git add src/descry/web/web/index.html
git commit -m "feat: add encyclopedia panel shell with TOC, search, and navigation"
```

---

### Task 4: Write tool articles (Search through Context — first 6)

**Files:**
- Modify: `src/descry/web/web/index.html` (inside the `ency-body` div, after the TOC block, replacing the `<!-- PLACEHOLDER -->` comment)

**Step 1: Add articles for Search, Semantic, Quick, Callers, Callees, Context**

Insert after the `<!-- Article Views -->` comment, before `</div><!-- ency-body -->`:

```html
<!-- ===== TOOL ARTICLES ===== -->

<!-- Search -->
<div class="ency-article" x-show="helpArticle === 'search'">
    <h2>Search</h2>
    <p class="ency-oneliner">Find symbols using hybrid keyword + semantic matching.</p>
    <h3>Parameters</h3>
    <table>
        <tr><th>Field</th><th>Description</th></tr>
        <tr><td>terms</td><td>Keywords to match against symbol names, docstrings, and code</td></tr>
        <tr><td>limit</td><td>Max results to return (default: 10)</td></tr>
        <tr><td>lang</td><td>Filter by language: python, rust, typescript, etc.</td></tr>
        <tr><td>package</td><td>Filter by top-level directory / package name</td></tr>
        <tr><td>type</td><td>Filter by symbol type: function, method, class, constant, file</td></tr>
    </table>
    <h3>Example</h3>
    <div class="ency-example">
        Search for <code>validate</code> &rarr; returns functions, methods, and classes whose name, docstring, or code contains "validate", ranked by relevance score (0&ndash;1).
    </div>
    <h3>Tips</h3>
    <ul class="ency-tips">
        <li>Combines keyword matching with semantic similarity when embeddings are available</li>
        <li>Click any result to jump to its full Context view</li>
        <li>For conceptual queries like "error handling middleware", try Semantic Search instead</li>
    </ul>
</div>

<!-- Semantic Search -->
<div class="ency-article" x-show="helpArticle === 'semantic'">
    <h2>Semantic Search</h2>
    <p class="ency-oneliner">Search by meaning using code embeddings rather than exact names.</p>
    <h3>Parameters</h3>
    <table>
        <tr><th>Field</th><th>Description</th></tr>
        <tr><td>query</td><td>Natural language description of what you're looking for</td></tr>
        <tr><td>limit</td><td>Max results to return (default: 10)</td></tr>
    </table>
    <h3>Example</h3>
    <div class="ency-example">
        Query <code>database connection pooling</code> &rarr; finds functions related to connection pools even if they're named <code>get_pool</code> or <code>acquire_conn</code>.
    </div>
    <h3>Tips</h3>
    <ul class="ency-tips">
        <li>Requires semantic embeddings to be built (check Health for availability)</li>
        <li>Best when you know what code does but not what it's called</li>
        <li>Use keyword Search when you know the exact symbol name</li>
    </ul>
</div>

<!-- Quick Lookup -->
<div class="ency-article" x-show="helpArticle === 'quick'">
    <h2>Quick Lookup</h2>
    <p class="ency-oneliner">Get full details on a symbol in a single step.</p>
    <h3>Parameters</h3>
    <table>
        <tr><th>Field</th><th>Description</th></tr>
        <tr><td>name</td><td>Symbol name to look up (e.g. function or class name)</td></tr>
        <tr><td>full</td><td>Show complete source without truncation</td></tr>
    </table>
    <h3>Example</h3>
    <div class="ency-example">
        Look up <code>parse_config</code> &rarr; returns source code, callers, callees, and related tests for the best-matching symbol, all in one response.
    </div>
    <h3>Tips</h3>
    <ul class="ency-tips">
        <li>Combines Search + Context into one step &mdash; fastest way to get a complete picture</li>
        <li>If the wrong match is found, use Search to pick the right one, then Context</li>
    </ul>
</div>

<!-- Callers -->
<div class="ency-article" x-show="helpArticle === 'callers'">
    <h2>Callers</h2>
    <p class="ency-oneliner">Find every function or method that calls the given symbol.</p>
    <h3>Parameters</h3>
    <table>
        <tr><th>Field</th><th>Description</th></tr>
        <tr><td>name</td><td>Symbol name to find callers of</td></tr>
        <tr><td>limit</td><td>Max callers to return (default: 20)</td></tr>
    </table>
    <h3>Example</h3>
    <div class="ency-example">
        Callers of <code>validate_token</code> &rarr; shows every function that invokes <code>validate_token</code>, with file locations and call count.
    </div>
    <h3>Tips</h3>
    <ul class="ency-tips">
        <li>More reliable than grep &mdash; uses the call graph to distinguish real calls from comments and strings</li>
        <li>Essential before refactoring: know who depends on a function before changing its signature</li>
        <li>Click any caller to view its full Context</li>
    </ul>
</div>

<!-- Callees -->
<div class="ency-article" x-show="helpArticle === 'callees'">
    <h2>Callees</h2>
    <p class="ency-oneliner">Find every function or method that the given symbol calls.</p>
    <h3>Parameters</h3>
    <table>
        <tr><th>Field</th><th>Description</th></tr>
        <tr><td>name</td><td>Symbol name to find callees of</td></tr>
        <tr><td>limit</td><td>Max callees to return (default: 20)</td></tr>
    </table>
    <h3>Example</h3>
    <div class="ency-example">
        Callees of <code>handle_request</code> &rarr; shows every function that <code>handle_request</code> invokes internally.
    </div>
    <h3>Tips</h3>
    <ul class="ency-tips">
        <li>Answers "what will break if this dependency changes?"</li>
        <li>Use Flow Trace for a recursive view (callees of callees of callees...)</li>
    </ul>
</div>

<!-- Context -->
<div class="ency-article" x-show="helpArticle === 'context'">
    <h2>Context</h2>
    <p class="ency-oneliner">Complete dossier for a symbol: source, callers, callees, tests.</p>
    <h3>Parameters</h3>
    <table>
        <tr><th>Field</th><th>Description</th></tr>
        <tr><td>node_id</td><td>Full node ID (e.g. <code>FILE:src/app.py::MyClass::init</code>)</td></tr>
        <tr><td>full</td><td>Show complete source without truncation</td></tr>
        <tr><td>expand callees</td><td>Inline source of small callees directly in the output</td></tr>
    </table>
    <h3>Example</h3>
    <div class="ency-example">
        Context for a function &rarr; shows its signature, docstring, full source code, list of callers, list of callees, and any test functions that reference it.
    </div>
    <h3>Tips</h3>
    <ul class="ency-tips">
        <li>Usually reached by clicking a result in Search, Callers, or other tools</li>
        <li>The node_id is the full graph identifier &mdash; you rarely need to type it manually</li>
        <li>"Expand callees" inlines the source of small helper functions so you can read them in place</li>
    </ul>
</div>
```

**Step 2: Verify articles are present**

Run:
```bash
cd /Users/christian/Documents/descry && python3 -c "
html = open('src/descry/web/web/index.html').read()
for slug in ['search', 'semantic', 'quick', 'callers', 'callees', 'context']:
    assert f\"helpArticle === '{slug}'\" in html, f'{slug} article not found'
print('First 6 tool articles verified OK')
"
```

**Step 3: Commit**

```bash
git add src/descry/web/web/index.html
git commit -m "content: add encyclopedia articles for Search through Context"
```

---

### Task 5: Write tool articles (Structure through Cross-Language — next 6)

**Files:**
- Modify: `src/descry/web/web/index.html` (append after Context article)

**Step 1: Add articles for Structure, Flatten, Impls, Flow, Path, Cross-Language**

```html
<!-- Structure -->
<div class="ency-article" x-show="helpArticle === 'structure'">
    <h2>Structure</h2>
    <p class="ency-oneliner">Show the skeleton of a file: all symbols with line numbers.</p>
    <h3>Parameters</h3>
    <table>
        <tr><th>Field</th><th>Description</th></tr>
        <tr><td>filename</td><td>File path or partial filename to match (e.g. <code>handlers.py</code>)</td></tr>
    </table>
    <h3>Example</h3>
    <div class="ency-example">
        Structure of <code>server.py</code> &rarr; lists imports, constants, classes, and functions defined in the file, with line numbers and types. Much faster than reading the full source.
    </div>
    <h3>Tips</h3>
    <ul class="ency-tips">
        <li>Great for orientation when entering an unfamiliar file</li>
        <li>Click any symbol to jump to its full Context</li>
        <li>Partial filenames work &mdash; <code>server</code> matches <code>web/server.py</code></li>
    </ul>
</div>

<!-- Flatten -->
<div class="ency-article" x-show="helpArticle === 'flatten'">
    <h2>Flatten</h2>
    <p class="ency-oneliner">Show the full public API of a type including inherited methods.</p>
    <h3>Parameters</h3>
    <table>
        <tr><th>Field</th><th>Description</th></tr>
        <tr><td>class_node_id</td><td>Full node ID of the class/struct (e.g. <code>FILE:src/app.py::MyClass</code>)</td></tr>
    </table>
    <h3>Example</h3>
    <div class="ency-example">
        Flatten <code>HttpClient</code> &rarr; shows all methods including those inherited from traits, interfaces, or parent classes, giving you the complete available API.
    </div>
    <h3>Tips</h3>
    <ul class="ency-tips">
        <li>Requires the full node ID &mdash; use Search or Structure to find it first</li>
        <li>Shows methods from all levels of the inheritance hierarchy</li>
    </ul>
</div>

<!-- Implementations -->
<div class="ency-article" x-show="helpArticle === 'impls'">
    <h2>Implementations</h2>
    <p class="ency-oneliner">Find all types that implement a trait or interface method.</p>
    <h3>Parameters</h3>
    <table>
        <tr><th>Field</th><th>Description</th></tr>
        <tr><td>method</td><td>Method name to find implementations of</td></tr>
        <tr><td>trait_name</td><td>Optional: filter to a specific trait or interface</td></tr>
    </table>
    <h3>Example</h3>
    <div class="ency-example">
        Impls of <code>parse</code> &rarr; shows every struct, class, or type that has a <code>parse</code> method, grouped by trait when applicable.
    </div>
    <h3>Tips</h3>
    <ul class="ency-tips">
        <li>Far more precise than grep for polymorphic code &mdash; filters out false positives</li>
        <li>Add a trait name to narrow results when a method name is common</li>
    </ul>
</div>

<!-- Flow Trace -->
<div class="ency-article" x-show="helpArticle === 'flow'">
    <h2>Flow Trace</h2>
    <p class="ency-oneliner">Visualize the recursive call tree from any symbol.</p>
    <h3>Parameters</h3>
    <table>
        <tr><th>Field</th><th>Description</th></tr>
        <tr><td>start</td><td>Symbol name to start tracing from</td></tr>
        <tr><td>direction</td><td><code>forward</code> (what it calls) or <code>backward</code> (what calls it)</td></tr>
        <tr><td>depth</td><td>How many levels deep to trace (default: 3)</td></tr>
        <tr><td>target</td><td>Optional: stop tracing when this symbol is reached</td></tr>
    </table>
    <h3>Example</h3>
    <div class="ency-example">
        Flow forward from <code>main</code>, depth 3 &rarr; shows a tree of <code>main</code> &rarr; its callees &rarr; their callees &rarr; their callees. Small functions are inlined with source code.
    </div>
    <h3>Tips</h3>
    <ul class="ency-tips">
        <li>Forward = "what happens when this runs?" / Backward = "how do we get here?"</li>
        <li>Click any node in the tree to jump to its Context</li>
        <li>Set a target to focus the trace on a specific call path</li>
        <li>Use Call Path instead if you just need the shortest route between two symbols</li>
    </ul>
</div>

<!-- Call Path -->
<div class="ency-article" x-show="helpArticle === 'path'">
    <h2>Call Path</h2>
    <p class="ency-oneliner">Find the shortest call chain between two symbols.</p>
    <h3>Parameters</h3>
    <table>
        <tr><th>Field</th><th>Description</th></tr>
        <tr><td>start</td><td>Source symbol name</td></tr>
        <tr><td>end</td><td>Target symbol name</td></tr>
        <tr><td>max_depth</td><td>Maximum hops to search (default: 10)</td></tr>
    </table>
    <h3>Example</h3>
    <div class="ency-example">
        Path from <code>handle_request</code> to <code>write_log</code> &rarr; shows each hop with the call site code snippet: <code>handle_request</code> &rarr; <code>process</code> &rarr; <code>write_log</code>.
    </div>
    <h3>Tips</h3>
    <ul class="ency-tips">
        <li>More focused than Flow Trace, which shows the entire tree</li>
        <li>Useful for verifying that a specific code path exists</li>
        <li>If no path is found, the symbols may be in disconnected parts of the graph</li>
    </ul>
</div>

<!-- Cross-Language -->
<div class="ency-article" x-show="helpArticle === 'cross-lang'">
    <h2>Cross-Language</h2>
    <p class="ency-oneliner">Trace API calls from frontend to backend via OpenAPI spec.</p>
    <h3>Parameters</h3>
    <table>
        <tr><th>Field</th><th>Description</th></tr>
        <tr><td>mode</td><td><code>endpoint</code> (single route), <code>list</code> (all routes for a tag), or <code>stats</code> (coverage overview)</td></tr>
        <tr><td>method</td><td>HTTP method: GET, POST, PUT, DELETE</td></tr>
        <tr><td>path</td><td>API path (e.g. <code>/api/v1/users</code>)</td></tr>
        <tr><td>tag</td><td>OpenAPI tag to filter by (for list mode)</td></tr>
    </table>
    <h3>Example</h3>
    <div class="ency-example">
        Endpoint <code>GET /api/v1/users</code> &rarr; shows which backend handler function serves this route and which frontend functions call it.
    </div>
    <h3>Tips</h3>
    <ul class="ency-tips">
        <li>Requires an OpenAPI spec to be configured (see project config)</li>
        <li>Use "stats" mode first to see available endpoints and coverage</li>
        <li>Bridges the gap when tracing user actions from UI to backend</li>
    </ul>
</div>
```

**Step 2: Verify articles are present**

Run:
```bash
cd /Users/christian/Documents/descry && python3 -c "
html = open('src/descry/web/web/index.html').read()
for slug in ['structure', 'flatten', 'impls', 'flow', 'path', 'cross-lang']:
    assert f\"helpArticle === '{slug}'\" in html, f'{slug} article not found'
print('Structure through Cross-Lang articles verified OK')
"
```

**Step 3: Commit**

```bash
git add src/descry/web/web/index.html
git commit -m "content: add encyclopedia articles for Structure through Cross-Language"
```

---

### Task 6: Write tool articles (Churn through Source — last 6)

**Files:**
- Modify: `src/descry/web/web/index.html` (append after Cross-Language article)

**Step 1: Add articles for Churn, Evolution, Changes, Health, Reindex, Source**

```html
<!-- Churn -->
<div class="ency-article" x-show="helpArticle === 'churn'">
    <h2>Churn</h2>
    <p class="ency-oneliner">Find the most frequently changed code using git history.</p>
    <h3>Parameters</h3>
    <table>
        <tr><th>Field</th><th>Description</th></tr>
        <tr><td>mode</td><td><code>symbols</code> (function-level), <code>files</code> (file-level), or <code>co-change</code> (coupled pairs)</td></tr>
        <tr><td>time_range</td><td>Git time range, e.g. <code>last 30 days</code>, <code>since 2024-01-01</code></td></tr>
        <tr><td>path_filter</td><td>Optional: limit to files matching a path prefix or pattern</td></tr>
        <tr><td>limit</td><td>Max results (default: 20)</td></tr>
    </table>
    <h3>Example</h3>
    <div class="ency-example">
        Churn in <code>symbols</code> mode, last 30 days &rarr; ranks functions by how many commits touched them. High-churn symbols may need refactoring or better test coverage.
    </div>
    <h3>Tips</h3>
    <ul class="ency-tips">
        <li><code>co-change</code> mode reveals hidden coupling: pairs of symbols that always change together</li>
        <li>Filter by path to focus on a specific module or directory</li>
        <li>Use Evolution to drill into a specific high-churn symbol's history</li>
    </ul>
</div>

<!-- Evolution -->
<div class="ency-article" x-show="helpArticle === 'evolution'">
    <h2>Evolution</h2>
    <p class="ency-oneliner">Track the commit history of a specific function or file.</p>
    <h3>Parameters</h3>
    <table>
        <tr><th>Field</th><th>Description</th></tr>
        <tr><td>name</td><td>Symbol or file name to track</td></tr>
        <tr><td>time_range</td><td>Optional: limit to a time window</td></tr>
        <tr><td>limit</td><td>Max commits to show (default: 10)</td></tr>
        <tr><td>show_diff</td><td>Include actual code diffs at each commit</td></tr>
    </table>
    <h3>Example</h3>
    <div class="ency-example">
        Evolution of <code>parse_config</code> with diffs &rarr; shows a timeline of commits, authors, dates, and the actual code changes at each point. Follows the function across renames.
    </div>
    <h3>Tips</h3>
    <ul class="ency-tips">
        <li>Uses git's native function tracking to follow symbols across line drift and renames</li>
        <li>Enable "Show diffs" to see what changed at each commit</li>
        <li>Great for understanding why a function looks the way it does</li>
    </ul>
</div>

<!-- Changes -->
<div class="ency-article" x-show="helpArticle === 'changes'">
    <h2>Changes</h2>
    <p class="ency-oneliner">Analyze recent commits to see modified symbols and their callers.</p>
    <h3>Parameters</h3>
    <table>
        <tr><th>Field</th><th>Description</th></tr>
        <tr><td>commit_range</td><td>Git range, e.g. <code>HEAD~5..HEAD</code>, <code>main..feature</code></td></tr>
        <tr><td>time_range</td><td>Alternative: time-based range like <code>last 7 days</code></td></tr>
        <tr><td>path_filter</td><td>Optional: limit to files matching a pattern</td></tr>
        <tr><td>show_callers</td><td>Include callers of changed functions (blast radius)</td></tr>
    </table>
    <h3>Example</h3>
    <div class="ency-example">
        Changes in <code>HEAD~5..HEAD</code> with callers &rarr; shows which functions were modified in the last 5 commits, plus every function that calls them, revealing the full blast radius.
    </div>
    <h3>Tips</h3>
    <ul class="ency-tips">
        <li>The "show callers" option is what makes this powerful &mdash; it reveals ripple risk</li>
        <li>Use for code review: see at a glance what a PR touches and what depends on it</li>
        <li>Defaults to the last commit if no range is specified</li>
    </ul>
</div>

<!-- Health -->
<div class="ency-article" x-show="helpArticle === 'health'">
    <h2>Health</h2>
    <p class="ency-oneliner">Check graph status, feature availability, and freshness.</p>
    <h3>Parameters</h3>
    <table>
        <tr><th>Field</th><th>Description</th></tr>
        <tr><td colspan="2"><em>No parameters &mdash; runs automatically</em></td></tr>
    </table>
    <h3>Example</h3>
    <div class="ency-example">
        Health &rarr; shows node count, edge count, graph age, and which features are available (SCIP indexing, semantic embeddings, git history, cross-language tracing).
    </div>
    <h3>Tips</h3>
    <ul class="ency-tips">
        <li>Check this first if tools return unexpected results &mdash; the graph may be stale</li>
        <li>The status dot in the header gives a quick summary (green = fresh, yellow = stale)</li>
    </ul>
</div>

<!-- Reindex -->
<div class="ency-article" x-show="helpArticle === 'reindex'">
    <h2>Reindex</h2>
    <p class="ency-oneliner">Rebuild the knowledge graph from scratch.</p>
    <h3>Parameters</h3>
    <table>
        <tr><th>Field</th><th>Description</th></tr>
        <tr><td colspan="2"><em>No parameters &mdash; rebuilds everything</em></td></tr>
    </table>
    <h3>Example</h3>
    <div class="ency-example">
        Reindex &rarr; re-scans all source files, rebuilds the call graph, regenerates SCIP indices and semantic embeddings. Progress streams live to the output log.
    </div>
    <h3>Tips</h3>
    <ul class="ency-tips">
        <li>Run after significant code changes: new files, renames, refactoring</li>
        <li>May take several minutes for large codebases with SCIP enabled</li>
        <li>Progress is streamed live &mdash; watch the log for current step</li>
    </ul>
</div>

<!-- Source -->
<div class="ency-article" x-show="helpArticle === 'source'">
    <h2>Source</h2>
    <p class="ency-oneliner">View source files with line numbers and optional line highlighting.</p>
    <h3>Parameters</h3>
    <table>
        <tr><th>Field</th><th>Description</th></tr>
        <tr><td>file</td><td>File path relative to project root</td></tr>
        <tr><td>line</td><td>Optional: line number to highlight and scroll to</td></tr>
    </table>
    <h3>Example</h3>
    <div class="ency-example">
        Source <code>src/app.py</code> line 42 &rarr; opens the file with line 42 highlighted. Useful for examining the surrounding code of a specific call site.
    </div>
    <h3>Tips</h3>
    <ul class="ency-tips">
        <li>Usually reached by clicking file locations in other tools</li>
        <li>Markdown files can be toggled between raw and rendered view</li>
    </ul>
</div>
```

**Step 2: Verify all 18 tool articles are present**

Run:
```bash
cd /Users/christian/Documents/descry && python3 -c "
html = open('src/descry/web/web/index.html').read()
tools = ['search', 'semantic', 'quick', 'callers', 'callees', 'context',
         'structure', 'flatten', 'impls', 'flow', 'path', 'cross-lang',
         'churn', 'evolution', 'changes', 'health', 'reindex', 'source']
for slug in tools:
    assert f\"helpArticle === '{slug}'\" in html, f'{slug} article missing'
print(f'All {len(tools)} tool articles verified OK')
"
```

**Step 3: Commit**

```bash
git add src/descry/web/web/index.html
git commit -m "content: add encyclopedia articles for Churn through Source"
```

---

### Task 7: Write concept articles

**Files:**
- Modify: `src/descry/web/web/index.html` (append after Source article)

**Step 1: Add 5 concept articles**

```html
<!-- ===== CONCEPT ARTICLES ===== -->

<!-- The Knowledge Graph -->
<div class="ency-article" x-show="helpArticle === 'concept-graph'">
    <h2>The Knowledge Graph</h2>
    <p class="ency-oneliner">How descry models your codebase as nodes and edges.</p>
    <h3>Nodes</h3>
    <p style="font-size:13px;color:var(--text-secondary);margin-bottom:8px">Every symbol in your code becomes a node: functions, methods, classes, constants, files, and configuration items. Each node has a unique ID like <code>FILE:src/app.py::MyClass::init</code> encoding its location in the file hierarchy.</p>
    <h3>Edges</h3>
    <p style="font-size:13px;color:var(--text-secondary);margin-bottom:8px">Edges represent relationships: <strong>calls</strong> (function A calls function B), <strong>contains</strong> (file contains class), <strong>inherits</strong> (class extends parent), and <strong>imports</strong>. This structure enables tools like Callers, Callees, Flow Trace, and Call Path.</p>
    <h3>How It's Built</h3>
    <p style="font-size:13px;color:var(--text-secondary)">Descry parses source files using language-aware AST parsers for Python, Rust, TypeScript, Go, Svelte, and more. When SCIP indexing is available, cross-references are precise (compiler-level accuracy). The graph is cached as JSON and rebuilt on demand via Reindex.</p>
</div>

<!-- Symbol Types -->
<div class="ency-article" x-show="helpArticle === 'concept-symbols'">
    <h2>Symbol Types</h2>
    <p class="ency-oneliner">The badge system used throughout descry results.</p>
    <h3>Legend</h3>
    <div>
        <div class="ency-legend-row"><span class="rc-type fun">FUN</span><span class="ency-legend-name">Function</span><span class="ency-legend-desc">Standalone functions and arrow functions</span></div>
        <div class="ency-legend-row"><span class="rc-type met">MET</span><span class="ency-legend-name">Method</span><span class="ency-legend-desc">Methods on a class, struct, or impl block</span></div>
        <div class="ency-legend-row"><span class="rc-type cla">CLA</span><span class="ency-legend-name">Class</span><span class="ency-legend-desc">Classes, structs, enums, traits, interfaces, protobuf messages</span></div>
        <div class="ency-legend-row"><span class="rc-type con">CON</span><span class="ency-legend-name">Constant</span><span class="ency-legend-desc">Exported constants and static values</span></div>
        <div class="ency-legend-row"><span class="rc-type fil">FIL</span><span class="ency-legend-name">File</span><span class="ency-legend-desc">Source files (.py, .rs, .ts, .svelte, .go, .proto, etc.)</span></div>
        <div class="ency-legend-row"><span class="rc-type cfg">CFG</span><span class="ency-legend-name">Configuration</span><span class="ency-legend-desc">Interceptors, middleware, and event handlers</span></div>
    </div>
    <h3>Tips</h3>
    <ul class="ency-tips">
        <li>The <strong>Score</strong> (0&ndash;1) shown in search results combines keyword and semantic relevance</li>
        <li>Click any badge in results to view that symbol's full Context</li>
        <li>Use the type filter in Search to narrow results to a specific kind of symbol</li>
    </ul>
</div>

<!-- SCIP Indexing -->
<div class="ency-article" x-show="helpArticle === 'concept-scip'">
    <h2>SCIP Indexing</h2>
    <p class="ency-oneliner">Compiler-level precision for cross-references and type resolution.</p>
    <h3>What is SCIP?</h3>
    <p style="font-size:13px;color:var(--text-secondary);margin-bottom:8px">SCIP (Source Code Intelligence Protocol) is a standard for representing code intelligence data. Descry uses SCIP indices produced by language-specific indexers (rust-analyzer for Rust, scip-typescript for TypeScript) to resolve cross-references with compiler-level accuracy.</p>
    <h3>Why It Matters</h3>
    <p style="font-size:13px;color:var(--text-secondary);margin-bottom:8px">Without SCIP, descry relies on AST-based name matching which can produce false positives (e.g., two functions with the same name in different modules). With SCIP, every reference is resolved to its exact definition &mdash; no ambiguity.</p>
    <h3>Supported Languages</h3>
    <p style="font-size:13px;color:var(--text-secondary)">Rust (via rust-analyzer), TypeScript/JavaScript (via scip-typescript). Check Health to see if SCIP is available for your project. SCIP is optional &mdash; all tools work without it, just with less precision.</p>
</div>

<!-- Semantic Embeddings -->
<div class="ency-article" x-show="helpArticle === 'concept-embeddings'">
    <h2>Semantic Embeddings</h2>
    <p class="ency-oneliner">How vector search finds conceptually similar code.</p>
    <h3>How It Works</h3>
    <p style="font-size:13px;color:var(--text-secondary);margin-bottom:8px">Each symbol's name, docstring, and signature are converted into a high-dimensional vector using a code embedding model. When you search, your query is also embedded, and results are ranked by cosine similarity &mdash; how close two vectors point in the same direction.</p>
    <h3>When to Use</h3>
    <p style="font-size:13px;color:var(--text-secondary);margin-bottom:8px"><strong>Semantic Search</strong>: when you know what code does but not its name. "rate limiting middleware" finds <code>throttle_requests</code>. <strong>Keyword Search</strong>: when you know the name. <code>throttle_requests</code> finds it directly.</p>
    <h3>Tips</h3>
    <ul class="ency-tips">
        <li>Embeddings are built during Reindex &mdash; check Health for availability</li>
        <li>The hybrid Search tool combines both approaches automatically</li>
    </ul>
</div>

<!-- Git History Analysis -->
<div class="ency-article" x-show="helpArticle === 'concept-git'">
    <h2>Git History Analysis</h2>
    <p class="ency-oneliner">How descry uses git log for churn, evolution, and change impact.</p>
    <h3>Three Perspectives</h3>
    <p style="font-size:13px;color:var(--text-secondary);margin-bottom:8px"><strong>Churn</strong> answers "what changes most?" by counting commits per symbol or file. <strong>Evolution</strong> answers "how did this change over time?" with a commit timeline. <strong>Changes</strong> answers "what did these commits affect?" with impact analysis.</p>
    <h3>How It Works</h3>
    <p style="font-size:13px;color:var(--text-secondary);margin-bottom:8px">Descry maps git diff output back to the knowledge graph. Changed line ranges are resolved to their enclosing functions, giving you symbol-level granularity instead of just file-level diffs.</p>
    <h3>Tips</h3>
    <ul class="ency-tips">
        <li>Requires a git repository &mdash; check Health for availability</li>
        <li>Time ranges use git's natural language: <code>last 30 days</code>, <code>since 2024-01-01</code></li>
    </ul>
</div>
```

**Step 2: Verify concept articles**

Run:
```bash
cd /Users/christian/Documents/descry && python3 -c "
html = open('src/descry/web/web/index.html').read()
for slug in ['concept-graph', 'concept-symbols', 'concept-scip', 'concept-embeddings', 'concept-git']:
    assert f\"helpArticle === '{slug}'\" in html, f'{slug} article missing'
print('All 5 concept articles verified OK')
"
```

**Step 3: Commit**

```bash
git add src/descry/web/web/index.html
git commit -m "content: add encyclopedia concept articles (graph, symbols, SCIP, embeddings, git)"
```

---

### Task 8: Write workflow articles

**Files:**
- Modify: `src/descry/web/web/index.html` (append after concept articles)

**Step 1: Add 3 workflow articles**

```html
<!-- ===== WORKFLOW ARTICLES ===== -->

<!-- Investigating a Function -->
<div class="ency-article" x-show="helpArticle === 'workflow-investigate'">
    <h2>Investigating a Function</h2>
    <p class="ency-oneliner">Start with a name, expand outward to build full understanding.</p>
    <h3>Steps</h3>
    <div class="ency-steps">
        <div class="ency-step"><span class="ency-step-num">1</span><span><strong>Quick Lookup</strong> the function name to get source, callers, callees, and tests in one shot.</span></div>
        <div class="ency-step"><span class="ency-step-num">2</span><span><strong>Context</strong> &mdash; click any related symbol to explore its details. Toggle "full" for complete source.</span></div>
        <div class="ency-step"><span class="ency-step-num">3</span><span><strong>Callers</strong> &mdash; understand who depends on this function and how it's used in practice.</span></div>
        <div class="ency-step"><span class="ency-step-num">4</span><span><strong>Flow Trace</strong> forward to see the full execution tree, or backward to see how control reaches this point.</span></div>
    </div>
    <h3>When to Use</h3>
    <ul class="ency-tips">
        <li>Before modifying a function &mdash; understand its callers and dependencies first</li>
        <li>When debugging &mdash; trace the flow to find where things go wrong</li>
        <li>During code review &mdash; verify that changes don't break callers</li>
    </ul>
</div>

<!-- Understanding a Codebase -->
<div class="ency-article" x-show="helpArticle === 'workflow-explore'">
    <h2>Understanding a Codebase</h2>
    <p class="ency-oneliner">Get the lay of the land when entering an unfamiliar project.</p>
    <h3>Steps</h3>
    <div class="ency-steps">
        <div class="ency-step"><span class="ency-step-num">1</span><span><strong>Health</strong> &mdash; see how large the codebase is (node/edge counts) and what features are available.</span></div>
        <div class="ency-step"><span class="ency-step-num">2</span><span><strong>Structure</strong> on key files (e.g. <code>main.py</code>, <code>app.rs</code>) to see what's defined where.</span></div>
        <div class="ency-step"><span class="ency-step-num">3</span><span><strong>Search</strong> for entry points like <code>main</code>, <code>handler</code>, <code>route</code> to find where execution starts.</span></div>
        <div class="ency-step"><span class="ency-step-num">4</span><span><strong>Flow Trace</strong> forward from entry points to understand the main execution paths.</span></div>
    </div>
    <h3>When to Use</h3>
    <ul class="ency-tips">
        <li>Onboarding onto a new project</li>
        <li>Returning to a codebase after a long break</li>
        <li>Evaluating an open-source library before adopting it</li>
    </ul>
</div>

<!-- Tracking Recent Changes -->
<div class="ency-article" x-show="helpArticle === 'workflow-changes'">
    <h2>Tracking Recent Changes</h2>
    <p class="ency-oneliner">See what's been happening in the codebase recently.</p>
    <h3>Steps</h3>
    <div class="ency-steps">
        <div class="ency-step"><span class="ency-step-num">1</span><span><strong>Changes</strong> with <code>HEAD~5..HEAD</code> and "show callers" to see recent modifications and their blast radius.</span></div>
        <div class="ency-step"><span class="ency-step-num">2</span><span><strong>Churn</strong> in symbols mode to identify which functions are changing most frequently.</span></div>
        <div class="ency-step"><span class="ency-step-num">3</span><span><strong>Evolution</strong> on high-churn symbols to see their full change timeline and understand patterns.</span></div>
    </div>
    <h3>When to Use</h3>
    <ul class="ency-tips">
        <li>Starting your workday &mdash; catch up on what teammates changed</li>
        <li>Before a release &mdash; assess the scope of recent changes</li>
        <li>Prioritizing refactoring &mdash; high-churn code benefits most from cleanup</li>
    </ul>
</div>
```

**Step 2: Verify all articles are present**

Run:
```bash
cd /Users/christian/Documents/descry && python3 -c "
html = open('src/descry/web/web/index.html').read()
all_slugs = [
    'search', 'semantic', 'quick', 'callers', 'callees', 'context',
    'structure', 'flatten', 'impls', 'flow', 'path', 'cross-lang',
    'churn', 'evolution', 'changes', 'health', 'reindex', 'source',
    'concept-graph', 'concept-symbols', 'concept-scip', 'concept-embeddings', 'concept-git',
    'workflow-investigate', 'workflow-explore', 'workflow-changes',
]
for slug in all_slugs:
    assert f\"helpArticle === '{slug}'\" in html, f'{slug} article missing'
print(f'All {len(all_slugs)} articles verified OK')
"
```
Expected: `All 26 articles verified OK`

**Step 3: Commit**

```bash
git add src/descry/web/web/index.html
git commit -m "content: add encyclopedia workflow articles (investigate, explore, track changes)"
```

---

### Task 9: End-to-end verification

**Files:**
- No files modified — verification only

**Step 1: Run the test suite**

Run: `cd /Users/christian/Documents/descry && python3 -m pytest tests/ -x -q`
Expected: All tests pass

**Step 2: Start the web server and verify the panel renders**

Run:
```bash
cd /Users/christian/Documents/descry && timeout 5 python3 -m descry.web.server --port 18923 &
sleep 2
# Check the index page loads
curl -s http://localhost:18923/ | python3 -c "
import sys
html = sys.stdin.read()
checks = [
    ('ency-overlay', 'Encyclopedia overlay'),
    ('ency-panel', 'Panel container'),
    ('ency-toc-item', 'TOC items'),
    ('ency-article', 'Article blocks'),
    ('helpArticle', 'Alpine article state'),
    ('openHelp()', 'Open help method'),
    ('concept-graph', 'Concept article'),
    ('workflow-investigate', 'Workflow article'),
]
ok = 0
for needle, label in checks:
    if needle in html:
        ok += 1
        print(f'  OK: {label}')
    else:
        print(f'  MISSING: {label}')
print(f'{ok}/{len(checks)} checks passed')
assert ok == len(checks), 'Some checks failed'
"
kill %1 2>/dev/null
```
Expected: `8/8 checks passed`

**Step 3: Verify article count matches TOC count**

Run:
```bash
cd /Users/christian/Documents/descry && python3 -c "
html = open('src/descry/web/web/index.html').read()
import re
# Count articles defined in helpArticles array
toc_count = html.count(\"{ slug: '\")
# Count article HTML blocks
article_count = len(re.findall(r'helpArticle === ', html))
# Subtract non-article uses (TOC click handlers etc) — each article has exactly 1 x-show
article_divs = len(re.findall(r'class=\"ency-article\" x-show', html))
print(f'TOC entries: {toc_count}')
print(f'Article divs: {article_divs}')
assert toc_count == article_divs, f'Mismatch: {toc_count} TOC entries vs {article_divs} article divs'
print('TOC and article counts match')
"
```
Expected: `TOC and article counts match`

**Step 4: Final commit (if any adjustments were needed)**

If all checks pass with no adjustments needed, no commit required. Otherwise, commit fixes.
