import json
import sys
from collections import defaultdict


def load_graph(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_skeleton(graph_file, output_file):
    data = load_graph(graph_file)

    # Organize by file
    files_content = defaultdict(list)
    file_nodes = {}

    for node in data["nodes"]:
        if node["type"] == "File":
            file_nodes[node["id"]] = node
        elif node["type"] in ("Class", "Function", "Method"):
            # Find parent file
            # ID format: FILE:path/to/file.py::Something
            file_id = node["id"].split("::")[0]
            files_content[file_id].append(node)

    # Sort files
    sorted_file_ids = sorted(file_nodes.keys())

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Codebase Skeleton\n\n")
        f.write(
            "Generated from static analysis. Contains signatures and docstrings only.\n\n"
        )

        for file_id in sorted_file_ids:
            file_node = file_nodes[file_id]
            file_path = file_node["metadata"]["path"]
            f.write(f"## File: `{file_path}`\n\n")

            # Sort children: Classes first, then Functions, by line number
            children = files_content[file_id]
            # Filter out the file node itself from children list if present
            children = [c for c in children if c["id"] != file_id]
            children.sort(key=lambda x: x["metadata"].get("lineno", 0))

            for child in children:
                meta = child["metadata"]
                name = meta.get("name", "???")
                node_type = child["type"]

                # Check nesting level
                # Simple heuristic: count '::'
                level = child["id"].count("::")
                indent = "  " * (level - 1) if level > 0 else ""

                if node_type == "Class":
                    f.write(f"{indent}- **class {name}**\n")
                    if meta.get("docstring"):
                        doc = meta["docstring"].split("\n")[0]  # First line only
                        f.write(f"{indent}  > {doc}\n")

                elif node_type in ("Function", "Method"):
                    sig = meta.get("signature", f"def {name}(...)")
                    f.write(f"{indent}- `{sig}`\n")
                    if meta.get("docstring"):
                        # Indent docstring
                        doc_lines = meta["docstring"].split("\n")
                        # Show max 3 lines
                        for line in doc_lines[:3]:
                            if line.strip():
                                f.write(f"{indent}  > {line.strip()}\n")
                        if len(doc_lines) > 3:
                            f.write(f"{indent}  > ...\n")

            f.write("\n---\n\n")

    print(f"Skeleton written to {output_file}")


if __name__ == "__main__":
    # Default paths use .descry_cache directory
    default_graph = ".descry_cache/codebase_graph.json"
    default_output = ".descry_cache/codebase_skeleton.md"

    graph_path = sys.argv[1] if len(sys.argv) > 1 else default_graph
    out_path = sys.argv[2] if len(sys.argv) > 2 else default_output
    generate_skeleton(graph_path, out_path)
