/**
 * Descry Tools Extension for Pi
 *
 * Registers descry tools as native pi tools that shell out to the
 * descry-cli.py wrapper via pi's safe execution API (pi.exec).
 * Also provides a /descry-setup command.
 *
 * Prerequisites:
 * - Python 3.11+
 * - descry package installed (pip install -e ~/Documents/descry)
 */

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Type } from "@sinclair/typebox";

const DESCRY_CLI = new URL("./descry-cli.py", import.meta.url).pathname;

export default function (pi: ExtensionAPI) {
	async function runDescry(
		subcommand: string,
		args: string[],
		signal?: AbortSignal,
	): Promise<string> {
		const fullArgs = [DESCRY_CLI, subcommand, ...args];
		const result = await pi.exec("python", fullArgs, {
			signal,
			timeout: 120_000,
		});

		if (result.code !== 0) {
			const error = result.stderr || result.stdout || "Unknown error";
			return JSON.stringify({ error, exitCode: result.code });
		}

		return result.stdout;
	}

	// --- /descry-setup command ---
	pi.registerCommand("descry-setup", {
		description: "Set up descry dependencies and verify installation",
		handler: async (_args, ctx) => {
			ctx.ui.notify("Checking descry prerequisites...", "info");

			const python = await pi.exec("python3", ["--version"], { timeout: 5000 });
			if (python.code !== 0) {
				ctx.ui.notify("Python 3 not found. Please install Python 3.11+.", "error");
				return;
			}
			ctx.ui.notify(`Found: ${python.stdout.trim()}`, "info");

			// Check if descry is installed
			const descryCheck = await pi.exec("python", ["-c", "import descry; print(descry.__version__)"], { timeout: 10000 });
			if (descryCheck.code !== 0) {
				ctx.ui.notify("descry not installed. Install with: pip install -e ~/Documents/descry", "error");
				return;
			}
			ctx.ui.notify(`Found: descry ${descryCheck.stdout.trim()}`, "info");

			ctx.ui.notify("Running descry health check...", "info");
			const health = await runDescry("health", []);
			ctx.ui.notify(`Descry setup complete. Health: ${health.slice(0, 200)}`, "success");
		},
	});

	// --- Descry tools ---

	pi.registerTool({
		name: "descry_ensure",
		label: "Descry Ensure",
		description:
			"Ensure the codebase graph exists and is fresh. Call this FIRST before other descry queries. " +
			"Regenerates if missing or older than max_age_hours. Returns graph status with node/edge counts. " +
			"WARNING: May take 30-60 seconds if regeneration is needed.",
		parameters: Type.Object({
			max_age_hours: Type.Optional(Type.Number({ description: "Max graph age in hours before regeneration", default: 24 })),
		}),
		async execute(_toolCallId, params, signal) {
			const args: string[] = [];
			if (params.max_age_hours !== undefined) args.push("--max-age-hours", String(params.max_age_hours));
			const result = await runDescry("ensure", args, signal);
			return { content: [{ type: "text", text: result }] };
		},
	});

	pi.registerTool({
		name: "descry_search",
		label: "Descry Search",
		description:
			"Search symbol names and docstrings. Returns compact single-line results by default. " +
			"Combines keyword + semantic search. Use filters for specific languages. " +
			"After finding candidates, use descry_context with brief=true to verify relevance.",
		parameters: Type.Object({
			query: Type.String({ description: "Search terms (space-separated)" }),
			compact: Type.Optional(Type.Boolean({ description: "Compact single-line output", default: true })),
			limit: Type.Optional(Type.Number({ description: "Max results", default: 10 })),
			lang: Type.Optional(Type.String({ description: "Filter by language (e.g., rust, python, typescript)" })),
			crate: Type.Optional(Type.String({ description: "Filter by crate/package name" })),
			type: Type.Optional(Type.String({ description: "Filter by symbol type (function, class, method)" })),
			exclude_tests: Type.Optional(Type.Boolean({ description: "Exclude test files", default: false })),
		}),
		async execute(_toolCallId, params, signal) {
			const args = ["--query", params.query];
			if (params.compact === false) args.push("--no-compact");
			if (params.limit !== undefined) args.push("--limit", String(params.limit));
			if (params.lang) args.push("--lang", params.lang);
			if (params.crate) args.push("--crate", params.crate);
			if (params.type) args.push("--type", params.type);
			if (params.exclude_tests) args.push("--exclude-tests");
			const result = await runDescry("search", args, signal);
			return { content: [{ type: "text", text: result }] };
		},
	});

	pi.registerTool({
		name: "descry_callers",
		label: "Descry Callers",
		description:
			"Find all functions/methods that call a given symbol. More reliable than grep for call " +
			"relationships - distinguishes actual calls from definitions and comments. " +
			"Use for impact analysis before refactoring.",
		parameters: Type.Object({
			name: Type.String({ description: "Symbol name to find callers of" }),
			limit: Type.Optional(Type.Number({ description: "Max results", default: 20 })),
		}),
		async execute(_toolCallId, params, signal) {
			const args = ["--symbol", params.name];
			if (params.limit !== undefined) args.push("--limit", String(params.limit));
			const result = await runDescry("callers", args, signal);
			return { content: [{ type: "text", text: result }] };
		},
	});

	pi.registerTool({
		name: "descry_callees",
		label: "Descry Callees",
		description:
			"Find what functions/methods a given symbol calls. Use for dependency analysis " +
			"and understanding what a function relies on.",
		parameters: Type.Object({
			name: Type.String({ description: "Symbol name to find callees of" }),
			limit: Type.Optional(Type.Number({ description: "Max results", default: 20 })),
		}),
		async execute(_toolCallId, params, signal) {
			const args = ["--symbol", params.name];
			if (params.limit !== undefined) args.push("--limit", String(params.limit));
			const result = await runDescry("callees", args, signal);
			return { content: [{ type: "text", text: result }] };
		},
	});

	pi.registerTool({
		name: "descry_context",
		label: "Descry Context",
		description:
			"Get full context for a symbol: source code, callers, callees, tests (~500+ tokens). " +
			"Use brief=true (~50 tokens) to quickly verify a symbol. " +
			"Options: full=true (no truncation), expand_callees=true (inline dependencies).",
		parameters: Type.Object({
			node_id: Type.String({ description: "Node ID from search results" }),
			brief: Type.Optional(Type.Boolean({ description: "Brief output (~50 tokens)", default: false })),
			full: Type.Optional(Type.Boolean({ description: "Full output (no truncation)", default: false })),
			expand_callees: Type.Optional(Type.Boolean({ description: "Inline callee source code", default: false })),
			deduplicate: Type.Optional(Type.Boolean({ description: "Skip recently shown content", default: false })),
			depth: Type.Optional(Type.Number({ description: "Traversal depth", default: 1 })),
			max_tokens: Type.Optional(Type.Number({ description: "Max output tokens", default: 2000 })),
		}),
		async execute(_toolCallId, params, signal) {
			const args = ["--node-id", params.node_id];
			if (params.brief) args.push("--brief");
			if (params.full) args.push("--full");
			if (params.expand_callees) args.push("--expand-callees");
			if (params.deduplicate) args.push("--deduplicate");
			if (params.depth !== undefined) args.push("--depth", String(params.depth));
			if (params.max_tokens !== undefined) args.push("--max-tokens", String(params.max_tokens));
			const result = await runDescry("context", args, signal);
			return { content: [{ type: "text", text: result }] };
		},
	});

	pi.registerTool({
		name: "descry_structure",
		label: "Descry Structure",
		description:
			"Show the structure of a file: imports, constants, classes, functions. " +
			"Faster than reading the entire file when you just need an overview.",
		parameters: Type.Object({
			path: Type.String({ description: "File path to analyze" }),
		}),
		async execute(_toolCallId, params, signal) {
			const result = await runDescry("structure", ["--path", params.path], signal);
			return { content: [{ type: "text", text: result }] };
		},
	});

	pi.registerTool({
		name: "descry_cross_lang",
		label: "Descry Cross-Language",
		description:
			"Trace API calls from frontend to backend handlers via OpenAPI spec. " +
			"Maps frontend API calls to backend implementations. " +
			"Use 'endpoint' mode to find which handler serves a specific endpoint.",
		parameters: Type.Object({
			mode: Type.Optional(Type.String({ description: "Mode: 'endpoint', 'list', or 'stats'", default: "endpoint" })),
			method: Type.Optional(Type.String({ description: "HTTP method (GET, POST, etc.)" })),
			path: Type.Optional(Type.String({ description: "API path to trace" })),
			tag: Type.Optional(Type.String({ description: "Resource tag to filter by" })),
		}),
		async execute(_toolCallId, params, signal) {
			const args: string[] = [];
			if (params.mode) args.push("--mode", params.mode);
			if (params.method) args.push("--method", params.method);
			if (params.path) args.push("--path", params.path);
			if (params.tag) args.push("--tag", params.tag);
			const result = await runDescry("cross_lang", args, signal);
			return { content: [{ type: "text", text: result }] };
		},
	});

	pi.registerTool({
		name: "descry_churn",
		label: "Descry Churn",
		description:
			"Find code churn hotspots - symbols or files that change most often. " +
			"Use for identifying unstable code, refactoring targets, or areas needing better tests. " +
			"Mode 'symbols' maps changes to functions, 'files' shows file-level stats, " +
			"'co-change' shows symbol pairs that frequently change together.",
		parameters: Type.Object({
			time_range: Type.Optional(Type.String({ description: "Git time range (e.g., '3 months')" })),
			path_filter: Type.Optional(Type.String({ description: "Filter by path prefix" })),
			limit: Type.Optional(Type.Number({ description: "Max results", default: 20 })),
			mode: Type.Optional(Type.String({ description: "Mode: 'symbols', 'files', or 'co-change'", default: "symbols" })),
		}),
		async execute(_toolCallId, params, signal) {
			const args: string[] = [];
			if (params.time_range) args.push("--time-range", params.time_range);
			if (params.path_filter) args.push("--path-filter", params.path_filter);
			if (params.limit !== undefined) args.push("--limit", String(params.limit));
			if (params.mode) args.push("--mode", params.mode);
			const result = await runDescry("churn", args, signal);
			return { content: [{ type: "text", text: result }] };
		},
	});

	pi.registerTool({
		name: "descry_quick",
		label: "Descry Quick",
		description:
			"Quickly find a symbol and show its full context in one step. " +
			"Combines search + context lookup - saves a round trip. " +
			"Set brief=true for minimal output, full=true for complete source.",
		parameters: Type.Object({
			name: Type.String({ description: "Symbol name to find" }),
			full: Type.Optional(Type.Boolean({ description: "Show complete source", default: false })),
			brief: Type.Optional(Type.Boolean({ description: "Minimal output", default: false })),
		}),
		async execute(_toolCallId, params, signal) {
			const args = ["--name", params.name];
			if (params.full) args.push("--full");
			if (params.brief) args.push("--brief");
			const result = await runDescry("quick", args, signal);
			return { content: [{ type: "text", text: result }] };
		},
	});

	pi.registerTool({
		name: "descry_health",
		label: "Descry Health",
		description:
			"Quick health check. Returns server version, graph status, and feature availability " +
			"(SCIP, embeddings). Use to verify connection and diagnose issues.",
		parameters: Type.Object({}),
		async execute(_toolCallId, _params, signal) {
			const result = await runDescry("health", [], signal);
			return { content: [{ type: "text", text: result }] };
		},
	});

	pi.registerTool({
		name: "descry_index",
		label: "Descry Index",
		description:
			"Regenerate the codebase graph, SCIP indices, and semantic embeddings. " +
			"Run after significant code changes (new files, refactoring, renamed symbols).",
		parameters: Type.Object({
			path: Type.Optional(Type.String({ description: "Path to index", default: "." })),
		}),
		async execute(_toolCallId, params, signal) {
			const args: string[] = [];
			if (params.path) args.push("--path", params.path);
			const result = await runDescry("index", args, signal);
			return { content: [{ type: "text", text: result }] };
		},
	});

	pi.registerTool({
		name: "descry_impls",
		label: "Descry Implementations",
		description:
			"Find all implementations of a trait method across the codebase. " +
			"Use when you know a trait method name but need to find which types implement it. " +
			"Optionally filter by specific trait name.",
		parameters: Type.Object({
			method: Type.String({ description: "Trait method name to find implementations of" }),
			trait_name: Type.Optional(Type.String({ description: "Filter by specific trait name" })),
		}),
		async execute(_toolCallId, params, signal) {
			const args = ["--method", params.method];
			if (params.trait_name) args.push("--trait-name", params.trait_name);
			const result = await runDescry("impls", args, signal);
			return { content: [{ type: "text", text: result }] };
		},
	});

	pi.registerTool({
		name: "descry_evolution",
		label: "Descry Evolution",
		description:
			"Track how a specific symbol has changed over time. " +
			"Shows commit timeline with authors and change sizes. " +
			"Set show_diff=true to include actual diff hunks.",
		parameters: Type.Object({
			name: Type.String({ description: "Symbol name to track" }),
			time_range: Type.Optional(Type.String({ description: "Git time range" })),
			limit: Type.Optional(Type.Number({ description: "Max commits", default: 10 })),
			show_diff: Type.Optional(Type.Boolean({ description: "Include diff hunks", default: false })),
			crate: Type.Optional(Type.String({ description: "Filter by crate/package name" })),
		}),
		async execute(_toolCallId, params, signal) {
			const args = ["--name", params.name];
			if (params.time_range) args.push("--time-range", params.time_range);
			if (params.limit !== undefined) args.push("--limit", String(params.limit));
			if (params.show_diff) args.push("--show-diff");
			if (params.crate) args.push("--crate", params.crate);
			const result = await runDescry("evolution", args, signal);
			return { content: [{ type: "text", text: result }] };
		},
	});

	pi.registerTool({
		name: "descry_changes",
		label: "Descry Changes",
		description:
			"Analyze change impact for a commit range. Maps changed lines to symbols " +
			"and shows their callers for ripple-risk assessment. " +
			"Defaults to HEAD~1..HEAD if no range specified.",
		parameters: Type.Object({
			commit_range: Type.Optional(Type.String({ description: "Git commit range (e.g., 'HEAD~5..HEAD')" })),
			time_range: Type.Optional(Type.String({ description: "Git time range" })),
			path_filter: Type.Optional(Type.String({ description: "Filter by path prefix" })),
			show_callers: Type.Optional(Type.Boolean({ description: "Show callers of changed symbols", default: true })),
			limit: Type.Optional(Type.Number({ description: "Max results", default: 50 })),
		}),
		async execute(_toolCallId, params, signal) {
			const args: string[] = [];
			if (params.commit_range) args.push("--commit-range", params.commit_range);
			if (params.time_range) args.push("--time-range", params.time_range);
			if (params.path_filter) args.push("--path-filter", params.path_filter);
			if (params.show_callers === false) args.push("--no-show-callers");
			if (params.limit !== undefined) args.push("--limit", String(params.limit));
			const result = await runDescry("changes", args, signal);
			return { content: [{ type: "text", text: result }] };
		},
	});

	pi.registerTool({
		name: "descry_flow",
		label: "Descry Flow",
		description:
			"Trace call flow from a starting symbol. Shows call chains with inline code " +
			"for small functions. Use for 'how does X reach Y' queries and impact analysis.",
		parameters: Type.Object({
			start: Type.String({ description: "Starting symbol name" }),
			direction: Type.Optional(Type.String({ description: "'forward' (callees) or 'backward' (callers)", default: "forward" })),
			depth: Type.Optional(Type.Number({ description: "Max traversal depth", default: 3 })),
			target: Type.Optional(Type.String({ description: "Stop when reaching this symbol" })),
		}),
		async execute(_toolCallId, params, signal) {
			const args = ["--start", params.start];
			if (params.direction) args.push("--direction", params.direction);
			if (params.depth !== undefined) args.push("--depth", String(params.depth));
			if (params.target) args.push("--target", params.target);
			const result = await runDescry("flow", args, signal);
			return { content: [{ type: "text", text: result }] };
		},
	});

	pi.registerTool({
		name: "descry_path",
		label: "Descry Path",
		description:
			"Find the shortest call path between two symbols. Shows each hop with " +
			"the call site code snippet. More focused than descry_flow.",
		parameters: Type.Object({
			start: Type.String({ description: "Starting symbol name" }),
			end: Type.String({ description: "Target symbol name" }),
			max_depth: Type.Optional(Type.Number({ description: "Max hops to search", default: 10 })),
			direction: Type.Optional(Type.String({ description: "'forward' or 'backward'", default: "forward" })),
		}),
		async execute(_toolCallId, params, signal) {
			const args = ["--start", params.start, "--end", params.end];
			if (params.max_depth !== undefined) args.push("--max-depth", String(params.max_depth));
			if (params.direction) args.push("--direction", params.direction);
			const result = await runDescry("path", args, signal);
			return { content: [{ type: "text", text: result }] };
		},
	});

	pi.registerTool({
		name: "descry_flatten",
		label: "Descry Flatten",
		description:
			"Show the effective API of a class including inherited methods. " +
			"Use for understanding class hierarchies in OOP codebases.",
		parameters: Type.Object({
			class_node_id: Type.String({ description: "Class node ID from search results" }),
		}),
		async execute(_toolCallId, params, signal) {
			const result = await runDescry("flatten", ["--node-id", params.class_node_id], signal);
			return { content: [{ type: "text", text: result }] };
		},
	});

	pi.registerTool({
		name: "descry_semantic",
		label: "Descry Semantic Search",
		description:
			"Pure semantic search using embeddings only (no keyword matching). " +
			"For most queries, use descry_search which intelligently combines both methods. " +
			"Requires sentence-transformers (optional dependency).",
		parameters: Type.Object({
			query: Type.String({ description: "Natural language search query" }),
			limit: Type.Optional(Type.Number({ description: "Max results", default: 10 })),
		}),
		async execute(_toolCallId, params, signal) {
			const args = ["--query", params.query];
			if (params.limit !== undefined) args.push("--limit", String(params.limit));
			const result = await runDescry("semantic", args, signal);
			return { content: [{ type: "text", text: result }] };
		},
	});
}
