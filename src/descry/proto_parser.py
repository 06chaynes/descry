"""The Protocol Buffers parser."""

import re

from descry._parsing import BaseParser, source_lines


class ProtoParser(BaseParser):
    def parse(self, _file_path, rel_path, content):
        file_id = self.add_file_node(rel_path, content)
        if package := re.search(
            r"(?m)^\s*package\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*;", content
        ):
            self.set_file_metadata(file_id, "proto_package", package[1])

        lines = source_lines(content)
        current_context = [file_id]
        context_indent = [0]

        re_type = re.compile(r"^\s*(?:service|message)\s+([a-zA-Z0-9_]+)")
        re_rpc = re.compile(
            r"^\s*rpc\s+([a-zA-Z0-9_]+)\s*\((.*?)\)\s*returns\s*\((.*?)\)"
        )

        for i, line in enumerate(lines):
            lineno = i + 1
            indent = len(line) - len(line.lstrip())

            if "}" in line:
                self.leave_contexts(current_context, context_indent, indent)

            parent_id = current_context[-1]

            if match := re_type.search(line):
                name = match.group(1)
                docstring = self.get_leading_docstring(lines, i)
                type_id = f"{parent_id}::{name}"
                self.builder.add_node(
                    type_id, "Class", name=name, lineno=lineno, docstring=docstring
                )
                self.builder.add_edge(parent_id, type_id, "DEFINES")
                if line.count("{") > line.count("}"):
                    current_context.append(type_id)
                    context_indent.append(indent)

            elif match := re_rpc.search(line):
                # A streamed message is the message.
                name, req, res = (
                    re.sub(r"^\s*stream\s+", "", g) for g in match.groups()
                )
                docstring = self.get_leading_docstring(lines, i)
                rid = f"{parent_id}::{name}"
                sig = f"rpc {name}({req}) returns ({res})"
                self.builder.add_node(
                    rid,
                    "Function",
                    name=name,
                    signature=sig,
                    lineno=lineno,
                    docstring=docstring,
                )
                self.builder.add_edge(parent_id, rid, "DEFINES")
                for message in filter(None, (req, res)):
                    self.builder.add_edge(
                        rid, f"REF:{message}", "CALLS", {"lineno": lineno}
                    )
