"""The Python parser."""

import ast
import re
from bisect import bisect_left, bisect_right, insort
from collections import deque
from operator import itemgetter

from descry import _stdlib
from descry._parsing import (
    BaseParser,
    SymbolTable,
    _apart,
    _bracketed_parts,
    _code_units,
    _disagree,
    _scip_columns,
    logger,
    merge_import,
    source_lines,
    token_count,
)

_PY_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _py_start(node):
    """The line and column ``node`` starts at, its decorators included."""
    first = (getattr(node, "decorator_list", None) or [node])[0]
    return first.lineno, first.col_offset


def _py_end(node):
    """The line and column ``node`` ends at."""
    return node.end_lineno, node.end_col_offset


# Compound statements whose blocks can hold definitions and imports.
_PY_COMPOUND_STMTS = (
    ast.If,
    ast.Try,
    ast.TryStar,
    ast.With,
    ast.AsyncWith,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Match,
)


def _py_block_statements(node):
    """Yield the statements in every block of a compound statement."""
    for block in ("body", "orelse", "finalbody"):
        yield from getattr(node, block, ())
    for handler in getattr(node, "handlers", ()):
        yield from handler.body
    for case in getattr(node, "cases", ()):
        yield from case.body


def _py_breaks(body):
    """Whether a `break` the statements in ``body`` run leaves the loop
    they are the body of; one inside a nested loop leaves that loop."""
    todo = list(body)
    while todo:
        stmt = todo.pop()
        if isinstance(stmt, ast.Break):
            return True
        if not isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            todo += [
                s
                for s in ast.iter_child_nodes(stmt)
                if isinstance(s, (ast.stmt, ast.match_case, ast.ExceptHandler))
            ]
    return False


def _py_ways(node):
    """The ways through ``node``, a compound statement that runs a body
    only where its head allows, as the statements each runs; an empty one
    runs none. None for any other statement."""
    if isinstance(node, ast.If):
        return [node.body, node.orelse]
    if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
        ways = [node.orelse, node.body + node.orelse]
        if _py_breaks(node.body):
            # A `break` leaves the loop before its `else` runs.
            ways.append(node.body)
        return ways
    if isinstance(node, (ast.Try, ast.TryStar)) and node.handlers:
        return [node.body + node.orelse, *(h.body for h in node.handlers)]
    if isinstance(node, ast.Match):
        return [*(case.body for case in node.cases), []]
    return None


_PY_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


# Nodes whose `name` they bind: `except E as name`, `case ... as name`, `*name`.
_PY_NAME_BINDERS = (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)


# The nodes ``PythonParser._push_bindings`` reads a binding, a narrowing or
# a returned value from
_PY_BINDING_NODES = (
    *_PY_DEFINITIONS,
    *_PY_COMPREHENSIONS,
    *_PY_NAME_BINDERS,
    ast.Assign,
    ast.AnnAssign,
    ast.NamedExpr,
    ast.Import,
    ast.ImportFrom,
    ast.For,
    ast.AsyncFor,
    ast.With,
    ast.AsyncWith,
    ast.Lambda,
    ast.If,
    ast.IfExp,
    ast.While,
    ast.Assert,
    ast.Match,
    ast.Return,
    ast.Yield,
    ast.YieldFrom,
)


# The builtin type of what each literal and display builds
_PY_LITERALS = {
    ast.JoinedStr: "str",
    ast.List: "list",
    ast.ListComp: "list",
    ast.Dict: "dict",
    ast.DictComp: "dict",
    ast.Set: "set",
    ast.SetComp: "set",
    ast.Tuple: "tuple",
}


# The method a binary operator calls on its left operand
_PY_OPERATORS = {
    ast.Add: "__add__",
    ast.Sub: "__sub__",
    ast.Mult: "__mul__",
    ast.MatMult: "__matmul__",
    ast.Div: "__truediv__",
    ast.FloorDiv: "__floordiv__",
    ast.Mod: "__mod__",
    ast.Pow: "__pow__",
    ast.LShift: "__lshift__",
    ast.RShift: "__rshift__",
    ast.BitOr: "__or__",
    ast.BitXor: "__xor__",
    ast.BitAnd: "__and__",
}


def _py_attribute(origin):
    """Whether the decorator of origin ``origin`` makes a method an
    attribute: it is a data descriptor of the library."""
    named = origin[1:] if origin.startswith("@") else f"builtins.{origin}"
    return named in _stdlib.section("python", "attributes")


def _py_import_bindings(node):
    """Yield ``(local name, dotted target)`` for each name an import statement
    binds. A relative target keeps its leading dots, and a star import is
    yielded under its own target, `pkg.mod.*`."""
    for alias in node.names:
        if isinstance(node, ast.Import):
            root = alias.name.split(".")[0]
            yield (alias.asname, alias.name) if alias.asname else (root, root)
        else:
            module = "." * node.level + (node.module or "")
            sep = "." if node.module else ""
            target = f"{module}{sep}{alias.name}"
            name = target if alias.name == "*" else alias.asname or alias.name
            yield name, target


def _py_strings(node):
    """The strings the list or tuple ``node`` writes, where it writes only
    strings; None otherwise."""
    if not isinstance(node, ast.List | ast.Tuple):
        return None
    names = [e.value for e in node.elts if isinstance(e, ast.Constant)]
    written = len(names) == len(node.elts)
    return names if written and all(isinstance(n, str) for n in names) else None


def _py_all_change(node):
    """How ``node`` changes `__all__`: ``("set", names)`` or ``("add",
    names)``, ``names`` None where they are not strings written out; None
    where it does not write `__all__`."""
    if isinstance(node, ast.Assign) and "__all__" in map(ast.unparse, node.targets):
        return "set", _py_strings(node.value)
    if isinstance(node, ast.AnnAssign | ast.AugAssign) and (
        ast.unparse(node.target) == "__all__" and node.value
    ):
        if isinstance(node, ast.AnnAssign):
            return "set", _py_strings(node.value)
        return "add", _py_strings(node.value) if isinstance(node.op, ast.Add) else None
    if (
        isinstance(node, ast.Delete) and "__all__" in map(ast.unparse, node.targets)
    ) or (
        # A slice of it stored, or the name written as a string:
        # `globals()["__all__"]`, `setattr(module, "__all__", …)`
        isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Store | ast.Del)
        and ast.unparse(node.value) == "__all__"
        or isinstance(node, ast.Constant)
        and node.value == "__all__"
    ):
        return "set", None
    method = isinstance(node, ast.Call) and node.func
    if isinstance(method, ast.Attribute) and ast.unparse(method.value) == "__all__":
        if method.attr == "append":
            return "add", _py_strings(ast.List(node.args))
        if method.attr == "extend" and len(node.args) == 1:
            return "add", _py_strings(node.args[0])
        return "add", None
    return None


def _py_all(body):
    """``(written, names)``: whether the module statements ``body``, or the
    functions and classes they define, write `__all__`, and the names it
    then lists, None where a write is not a statement of the module of its
    own, run whatever the condition, or does not write the names out."""
    written, listed = False, []
    for statement in body:
        todo = deque([statement])
        while todo:
            node = todo.popleft()
            todo.extend(ast.iter_child_nodes(node))
            change = _py_all_change(node)
            if not change:
                continue
            written, (kind, names) = True, change
            alone = node is statement or getattr(statement, "value", None) is node
            if names is None or not alone:
                listed = None
            elif kind == "set":
                listed = names
            elif listed is not None:
                listed = listed + names
    return written, listed


def _py_is_none(node):
    """Whether ``node`` is `None`, which says nothing of what the name it
    is assigned to holds otherwise."""
    return isinstance(node, ast.Constant) and node.value is None


def _py_targets(target, origin):
    """Yield ``(name, origin)`` for each name an assignment of the value at
    ``origin`` to ``target`` binds: `@N` leads to item N of what is
    unpacked."""
    if isinstance(target, ast.Name):
        yield target.id, origin
    elif isinstance(target, (ast.Tuple, ast.List)):
        starred = False
        for position, element in enumerate(target.elts):
            starred = starred or isinstance(element, ast.Starred)
            yield from _py_targets(element, "?" if starred else f"{origin}.@{position}")
    elif isinstance(target, ast.Starred):
        yield from _py_targets(target.value, "?")


def _py_nested_definitions(statements):
    """Yield the definitions among ``statements``, looking through compound
    statements but not into other definitions."""
    for stmt in statements:
        if isinstance(stmt, _PY_DEFINITIONS):
            yield stmt
        elif isinstance(stmt, _PY_COMPOUND_STMTS):
            yield from _py_nested_definitions(_py_block_statements(stmt))


class PythonParser(BaseParser):
    def parse(self, file_path, rel_path, content):
        file_id = f"FILE:{rel_path}"
        self.builder.current_file_id = file_id
        self.builder.current_scope = SymbolTable()

        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            logger.warning(f"Syntax error in {rel_path}:{e.lineno}: {e.msg}")
            return
        except (ValueError, RecursionError) as e:
            logger.warning(f"Parse error in {rel_path}: {e}")
            return

        self.add_file_node(rel_path, content)
        self.source_lines = source_lines(content)
        self.class_ids = set()
        self.import_targets = {}
        # name -> "import" where the module's import of it rebinds its
        # definition, "either" where which one binds is not told
        self.import_binding = {}
        self.bindings = []
        # id() of a name read in a comprehension, a lambda or a narrowed
        # block -> origin of what it is bound to there
        self.scoped = {}
        # ``(start, end, ways)`` of each `if` and `match`: where it is and
        # the spans of the statements of each of its ways
        self.branches = []
        self._push_bindings(tree.body)
        self.builder.fields[file_id] = {
            name: sorted({origin for _, origin in bound})
            for name, bound in self.bindings[0].items()
        }
        self.visit_node(tree, file_id)
        self.builder.imports[file_id] = self.import_targets
        self.builder.import_binding[file_id] = self.import_binding

    def _get_type_annotation(self, annotation):
        """Extract type annotation as a string."""
        if annotation is None:
            return "Any"
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Attribute):
            base = self._get_attribute_name(annotation.value)
            return f"{base}.{annotation.attr}" if base else annotation.attr
        elif isinstance(annotation, ast.Subscript):
            base = self._get_type_annotation(annotation.value)
            slice_val = annotation.slice
            if isinstance(slice_val, ast.Tuple):
                elts = ", ".join(self._get_type_annotation(e) for e in slice_val.elts)
                return f"{base}[{elts}]"
            else:
                return f"{base}[{self._get_type_annotation(slice_val)}]"
        elif isinstance(annotation, ast.Constant):
            return (
                repr(annotation.value)
                if isinstance(annotation.value, str)
                else str(annotation.value)
            )
        elif isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            left = self._get_type_annotation(annotation.left)
            right = self._get_type_annotation(annotation.right)
            return f"{left} | {right}"
        elif isinstance(annotation, ast.Tuple):
            elts = ", ".join(self._get_type_annotation(e) for e in annotation.elts)
            return f"({elts})"
        elif isinstance(annotation, ast.List):
            elts = ", ".join(self._get_type_annotation(e) for e in annotation.elts)
            return f"[{elts}]"
        return "Complex"

    def _get_param_types(self, args_node):
        """Extract parameter types as a list for metadata storage."""
        return [self._get_type_annotation(arg.annotation) for arg in args_node.args]

    def _record_decorators(self, node, owner_id):
        """Record the call of each decorator of ``node`` from ``owner_id``."""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                self._record_call(decorator, owner_id, decorator.lineno)
            else:
                self._record_callee(decorator, owner_id, decorator.lineno)

    def visit_node(self, node, parent_id):
        if isinstance(node, ast.ClassDef):
            class_id = f"{parent_id}::{node.name}"
            self.builder.add_node(
                class_id,
                "Class",
                name=node.name,
                lineno=node.lineno,
                end_lineno=node.end_lineno,
                token_count=token_count(node.lineno, node.end_lineno),
                docstring=ast.get_docstring(node) or "",
            )
            self.builder.add_edge(parent_id, class_id, "DEFINES")
            self.class_ids.add(class_id)

            self._record_decorators(node, class_id)

            for base in node.bases:
                if isinstance(base, ast.Subscript):
                    base = base.value
                base_name = self._get_attribute_name(base)
                if base_name:
                    resolved_base = self.builder.current_scope.resolve(base_name)
                    self.builder.add_edge(
                        class_id,
                        f"REF:{resolved_base}",
                        "INHERITS",
                        {
                            "origin": self._origin(base, node.lineno),
                            # The base's own position: SCIP resolves the
                            # edge at its occurrence, not by bare name.
                            "lineno": base.lineno,
                            "col": base.col_offset,
                        },
                    )

            # The class binds its name for what is written after it.
            self.builder.current_scope.add_import(node.name)
            self.builder.fields[class_id] = self._class_attrs(node, class_id)
            # Each name the body binds last by an assignment, not a `def` or
            # a `class`: the function called on its `def` where that is the
            # value, `z = staticmethod(z)`, else None.
            assigned, defined = {}, set()
            for stmt in node.body:
                if isinstance(stmt, _PY_DEFINITIONS):
                    assigned.pop(stmt.name, None)
                    defined.add(stmt.name)
                elif isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value:
                    value = stmt.value
                    targets = getattr(stmt, "targets", None) or [stmt.target]
                    for target in targets:
                        for name in ast.walk(target):
                            if isinstance(name, ast.Name) and isinstance(
                                name.ctx, ast.Store
                            ):
                                wraps = (
                                    name.id in defined
                                    and name.id not in assigned
                                    and isinstance(value, ast.Call)
                                    and isinstance(value.func, ast.Name)
                                    and not value.keywords
                                    and len(value.args) == 1
                                    and isinstance(value.args[0], ast.Name)
                                    and value.args[0].id == name.id
                                )
                                assigned[name.id] = value.func.id if wraps else None
            self.builder.py_class_assigned[class_id] = assigned
            for child in node.body:
                self.visit_node(child, class_id)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_id = f"{parent_id}::{node.name}"
            # A function is a method iff its enclosing parent node is a Class.
            is_method = parent_id in self.class_ids
            node_type = "Method" if is_method else "Function"

            param_types = self._get_param_types(node.args)
            return_type = (
                self._get_type_annotation(node.returns) if node.returns else ""
            )
            keyword = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            signature = f"{keyword} {node.name}({ast.unparse(node.args)})"
            if node.returns:
                signature += f" -> {ast.unparse(node.returns)}"

            self.builder.add_node(
                func_id,
                node_type,
                name=node.name,
                signature=signature,
                lineno=node.lineno,
                end_lineno=node.end_lineno,
                token_count=token_count(node.lineno, node.end_lineno),
                docstring=ast.get_docstring(node) or "",
                return_type=return_type,
                param_types=param_types,
            )
            metadata = self.builder.nodes[-1]["metadata"]
            self.builder.add_edge(parent_id, func_id, "DEFINES")

            self._record_decorators(node, func_id)
            if node.decorator_list and not is_method:
                # The name is bound to what the outermost decorator returns.
                made = self._origin(node.decorator_list[0], node.lineno)
                self.builder.fields.setdefault(parent_id, {})[node.name] = [f"{made}()"]

            self._push_bindings(node.body, node, is_method)
            origins, yields = self._returned_origins()
            if isinstance(node, ast.AsyncFunctionDef) and not yields:
                # Calling it gives a coroutine, which `await` runs.
                metadata["coroutine"] = True
            fixture = self._fixture_name(node)
            if fixture:
                origin = origins[0] if len(origins) == 1 else None
                self.builder.fixtures[parent_id, fixture] = (func_id, origin, yields)
            # What calling the function gives, `()`, is what it returns
            # where it is no generator and neither an annotation nor a
            # decorator says otherwise. Under a context-manager decorator
            # the key is the decorator, for what the function yields.
            made = {self._origin(d, node.lineno) for d in node.decorator_list}
            # One that makes a generator function return a context manager,
            # which hands the `with` block what the generator yields
            managers = _stdlib.section("python", "context_managers")
            manager = next((m for m in managers if f"@{m}" in made), None)
            plain = not (
                node.returns or yields or made - {"staticmethod", "classmethod"}
            )
            # An async generator's item is what its awaited `__anext__` gives.
            step = (
                "__anext__().await"
                if isinstance(node, ast.AsyncFunctionDef)
                else "__next__()"
            )
            if manager and node.returns:
                origins = [f"{self._instance_origin(node.returns)}.{step}"]
            if origins and (plain or manager):
                held = self.builder.fields.setdefault(func_id, {})
                held[manager or "()"] = origins
            self.visit_body_for_calls(node.body, func_id)

            for child in _py_nested_definitions(node.body):
                self.visit_node(child, func_id)
            self.bindings.pop()

        elif isinstance(node, ast.Import):
            for alias in node.names:
                self.builder.current_scope.add_import(alias.name, alias=alias.asname)
                self.builder.add_edge(
                    self.builder.current_file_id, f"MODULE:{alias.name}", "IMPORTS"
                )

        elif isinstance(node, ast.ImportFrom):
            module = node.module if node.module else ""
            for alias in node.names:
                full_name = f"{module}.{alias.name}" if module else alias.name
                self.builder.current_scope.add_import(
                    alias.name, module=module, alias=alias.asname
                )
                self.builder.add_edge(
                    self.builder.current_file_id, f"MODULE:{full_name}", "IMPORTS"
                )

        elif isinstance(node, ast.Module):
            for child in node.body:
                self.visit_node(child, parent_id)

        elif isinstance(node, _PY_COMPOUND_STMTS):
            for child in _py_block_statements(node):
                self.visit_node(child, parent_id)

    def visit_body_for_calls(self, body, parent_func_id):
        """Find all Call nodes in the function body.

        This catches calls in all contexts: comprehensions, generators,
        with statements, if/while conditions, for iterables, decorators,
        nested calls, await expressions, etc. Nested function and class
        definitions are not entered: visit_node records their calls. The
        default arguments of a nested function are evaluated where it is
        defined, so their calls belong here.
        """
        todo = deque(body)
        while todo:
            node = todo.popleft()
            if isinstance(node, _PY_DEFINITIONS):
                if not isinstance(node, ast.ClassDef):
                    defaults = (*node.args.defaults, *node.args.kw_defaults)
                    todo.extend(default for default in defaults if default)
                continue
            todo.extend(ast.iter_child_nodes(node))
            if isinstance(node, ast.Call):
                self._record_call(node, parent_func_id, node.lineno)

    def _record_call(self, call_node, parent_func_id, lineno):
        index = len(self.builder.edges)
        self._record_callee(call_node.func, parent_func_id, lineno)
        if index == len(self.builder.edges):
            return
        # `#N|argument|P` is parameter P of the callable passed as that
        # argument to the call at edge index N.
        passed = [(str(at), arg) for at, arg in enumerate(call_node.args)]
        passed += [(keyword.arg, keyword.value) for keyword in call_node.keywords]
        for argument, value in passed:
            if isinstance(value, ast.Lambda):
                for position, arg in enumerate(value.args.args):
                    origin = f"#{index}|{argument}|{position}"
                    self._bind_scoped(value.body, ast.Name(arg.arg), origin)

    def _record_callee(self, func_node, parent_func_id, lineno):
        func_name = self._get_attribute_name(func_node)
        if func_name:
            resolved_name = self.builder.current_scope.resolve(func_name)
            origin = self._origin(func_node, lineno)
            if isinstance(func_node, ast.Name):
                # A variable that is called holds a value, `=name`, which
                # returns `Type` where it is annotated as a callable that
                # does: `=name>Type`.
                annotated = re.fullmatch(r":(?:\w+::)*Callable\[(.+)\]", origin)
                if origin == "?":
                    origin = f"={func_name}"
                elif annotated:
                    origin = f"={func_name}>{_bracketed_parts(annotated[1])[-1]}"
            metadata = {"lineno": lineno, "origin": origin}
            if func_node.end_lineno == lineno:
                # The called name ends where the callee expression does. It
                # is read from the line: the AST holds its NFKC form, which
                # can differ in length.
                line = self.source_lines[lineno - 1].encode()
                written = line[: func_node.end_col_offset].decode()
                name = re.search(r"\w*$", written)
                metadata["col"] = _scip_columns(_code_units(written[: name.start()]))
            self.builder.add_edge(
                parent_func_id, f"REF:{resolved_name}", "CALLS", metadata
            )

    def _origin(self, node, lineno):
        """Write the expression ``node`` as an origin path for CallResolver:
        dotted names with ``()`` after each call, imported names expanded
        behind ``@``, and local names replaced by what they were bound to
        before ``lineno``. A literal is a call of its builtin type, a
        subscript and a binary operator a call of the method they stand
        for. ``?`` stands for anything that has no such path."""
        if isinstance(node, ast.Name):
            if id(node) in self.scoped:
                return self.scoped[id(node)]
            read = getattr(node, "lineno", lineno), getattr(node, "col_offset", 0)
            for depth, frame in enumerate(reversed(self.bindings)):
                if node.id in frame:
                    bound = frame[node.id]
                    # The last made, of those of an outer frame, else of
                    # those made before the read on a way to it
                    if depth:
                        return bound[-1][1]
                    below = bisect_right(bound, read, key=itemgetter(0))
                    return next(
                        (
                            bound[at][1]
                            for at in reversed(range(below))
                            if not _apart(self.branches, bound[at][0], read)
                        ),
                        "?",
                    )
            imported = self.import_targets.get(node.id)
            return f"@{imported}" if isinstance(imported, str) else node.id
        if isinstance(node, ast.Attribute):
            return f"{self._origin(node.value, lineno)}.{node.attr}"
        if isinstance(node, ast.Call):
            return f"{self._origin(node.func, lineno)}()"
        if isinstance(node, ast.Await):
            return f"{self._origin(node.value, lineno)}.await"
        literal = _PY_LITERALS.get(type(node))
        if isinstance(node, ast.Constant) and node.value not in (None, ...):
            literal = type(node.value).__name__
        if literal:
            return f"@builtins.{literal}()"
        if isinstance(node, ast.Subscript) and not isinstance(node.slice, ast.Slice):
            return f"{self._origin(node.value, lineno)}.__getitem__()"
        if isinstance(node, ast.BinOp) and type(node.op) in _PY_OPERATORS:
            return f"{self._origin(node.left, lineno)}.{_PY_OPERATORS[type(node.op)]}()"
        return "?"

    def _fixture_name(self, func):
        """Name of the pytest fixture ``func`` is decorated as, if any."""
        for decorator in func.decorator_list:
            called = isinstance(decorator, ast.Call)
            target = decorator.func if called else decorator
            if self._origin(target, func.lineno) != "@pytest.fixture":
                continue
            for keyword in decorator.keywords if called else ():
                given = keyword.value
                if keyword.arg == "name" and isinstance(given, ast.Constant):
                    return str(given.value)
            return func.name
        return None

    def _returned_origins(self):
        """``(origins, yields)`` for the function whose bindings were pushed
        last: the origins of the values other than `None` that it returns or
        yields, none where one of them has no origin path, and whether it
        yields."""
        origins, yields = set(), False
        for node in self.exits:
            if isinstance(node, ast.YieldFrom):
                iterated = self._origin(node.value, node.lineno)
                origins.add(f"{iterated}.__iter__().__next__()")
            elif node.value and not _py_is_none(node.value):
                origins.add(self._origin(node.value, node.lineno))
            yields = yields or not isinstance(node, ast.Return)
        known = not any("?" in origin for origin in origins)
        return sorted(origins) if known else [], yields

    def _instance_origin(self, annotation):
        """Origin of a value annotated with ``annotation``: `:type`, the
        type as written with its dots as `::`."""
        written = self._get_type_annotation(annotation)
        if "(" in written or "Complex" in written:
            return "?"
        return ":" + written.replace(".", "::")

    def _bind_scoped(self, owner, target, origin):
        """Bind the names an assignment of the value at ``origin`` to
        ``target`` binds, where ``owner`` reads them."""
        bound = dict(_py_targets(target, origin))
        for read in ast.walk(owner):
            if isinstance(read, ast.Name) and read.id in bound:
                self.scoped[id(read)] = bound[read.id]

    def _push_bindings(self, body, func=None, is_method=False):
        """Push a frame mapping each name bound in ``body`` (and by the
        parameters of ``func``) to its ``((line, column), origin)``
        bindings, at where each takes effect: the end of what binds it, or
        the start of the body a parameter, a loop, a `with` or an `except`
        binds it for, in the order of where they take effect. Module-level
        imports go to ``import_targets``."""
        frame = {}
        self.bindings.append(frame)

        def put(name, at, origin):
            insort(frame.setdefault(name, []), (at, origin), key=itemgetter(0))

        def bind(target, at, origin):
            for name, bound in _py_targets(target, origin):
                put(name, at, bound)

        def iterated(value, loop, lineno):
            """Origin of what ``loop`` binds on each pass over ``value``."""
            step = "__iter__().__next__()"
            if isinstance(loop, ast.AsyncFor) or getattr(loop, "is_async", 0):
                step = "__aiter__().__anext__().await"
            return f"{self._origin(value, lineno)}.{step}"

        def checked(test):
            """Yield ``(name, origin of an instance of the class)`` for
            each `isinstance(name, Class)` that holds where ``test`` does."""
            conjunction = isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And)
            for check in test.values if conjunction else [test]:
                if (
                    isinstance(check, ast.Call)
                    and isinstance(check.func, ast.Name)
                    and check.func.id == "isinstance"
                    and len(check.args) == 2
                    and isinstance(check.args[0], ast.Name)
                ):
                    yield check.args[0], self._instance_origin(check.args[1])

        def narrow(blocks, name, origin):
            """Type ``name`` where ``blocks`` read it, unless they rebind
            it."""
            reads = [
                read
                for block in blocks
                for read in ast.walk(block)
                if isinstance(read, ast.Name) and read.id == name.id
            ]
            if "?" not in origin and all(isinstance(r.ctx, ast.Load) for r in reads):
                self.scoped.update((id(read), origin) for read in reads)

        if func is not None:
            args = func.args
            params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
            params += [a for a in (args.vararg, args.kwarg) if a]
            decorators = {d.id for d in func.decorator_list if isinstance(d, ast.Name)}
            positional = len(args.posonlyargs) + len(args.args)
            # pytest passes a test or fixture the fixtures its parameters
            # name, unless a `parametrize` mark supplies the argument.
            injected = func.name.startswith("test") or self._fixture_name(func)
            parametrized = {
                name.strip()
                for decorator in func.decorator_list
                if isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "parametrize"
                and decorator.args
                for constant in ast.walk(decorator.args[0])
                if isinstance(constant, ast.Constant)
                and isinstance(constant.value, str)
                for name in constant.value.split(",")
            }
            for index, arg in enumerate(params):
                if (
                    index == 0
                    and positional
                    and is_method
                    and "staticmethod" not in decorators
                ):
                    origin = "cls" if "classmethod" in decorators else "self"
                elif arg is args.vararg:
                    origin = "@builtins.tuple()"
                elif arg is args.kwarg:
                    origin = "@builtins.dict()"
                elif arg.annotation:
                    origin = self._instance_origin(arg.annotation)
                elif injected and arg.arg not in parametrized:
                    origin = f"~{arg.arg}"
                else:
                    origin = "?"
                frame[arg.arg] = [(_py_start(func.body[0]), origin)]

        found, branching = [], []
        # name -> ("def" or "import", whether a conditional holds it, where
        # it is written) of each binding of it at module level; a star
        # import, which may bind any name, is filed under `*`
        module_bound = {}

        def ways_of(node):
            """The ways of each conditional statement whose spans hold
            ``node``, as ``(statement, way)``: which it may run in."""
            start = _py_start(node)
            return {
                (at, way)
                for at, (_, _, _, spans) in enumerate(branching)
                for way, held in enumerate(spans)
                for span in held
                if span[0] <= start < span[1]
            }

        self.exits = []  # the `return` and `yield` nodes of ``body``
        todo = deque(body)
        while todo:
            node = todo.popleft()
            if isinstance(node, _PY_BINDING_NODES):
                found.append(node)
            if ways := _py_ways(node):
                spans = [[(_py_start(s), _py_end(s)) for s in way] for way in ways]
                first = min(span[0] for way in spans for span in way)
                branching.append((node, first, ways, spans))
                if isinstance(node, (ast.If, ast.Match)):
                    self.branches.append((first, _py_end(node), spans))
            if not isinstance(node, _PY_DEFINITIONS):
                todo.extend(ast.iter_child_nodes(node))
        for node in sorted(found, key=lambda n: n.lineno):
            lineno = node.lineno
            if isinstance(node, _PY_DEFINITIONS):
                frame.pop(node.name, None)
                # A `def` or `class` binds its name for what is written
                # after it, which shadows an import of the name.
                put(node.name, _py_end(node), node.name)
                if func is None:
                    module_bound.setdefault(node.name, []).append(
                        ("def", bool(ways_of(node)), _py_start(node))
                    )
            elif isinstance(node, ast.Assign) and not _py_is_none(node.value):
                origin = self._origin(node.value, lineno)
                for target in node.targets:
                    bind(target, _py_end(node), origin)
            elif isinstance(node, ast.AnnAssign):
                origin = self._instance_origin(node.annotation)
                bind(node.target, _py_end(node), origin)
            elif isinstance(node, ast.NamedExpr):
                bind(node.target, _py_end(node), self._origin(node.value, lineno))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for name, target in _py_import_bindings(node):
                    if func is None:
                        conditional = bool(ways_of(node))
                        module_bound.setdefault(
                            "*" if target.endswith(".*") else name, []
                        ).append(("import", conditional, _py_start(node)))
                        if conditional:
                            # An import a conditional holds and another of
                            # its ways may bind: which one is not told. It
                            # merges with the targets already there.
                            merge_import(self.import_targets, name, target)
                        else:
                            # An unconditional import rebinds the name:
                            # the later one is what binds after it.
                            self.import_targets[name] = target
                    else:
                        put(name, _py_end(node), f"@{target}")
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                origin = iterated(node.iter, node, lineno)
                bind(node.target, _py_start(node.body[0]), origin)
            elif isinstance(node, _PY_COMPREHENSIONS):
                for generator in node.generators:
                    origin = iterated(generator.iter, generator, lineno)
                    self._bind_scoped(node, generator.target, origin)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                # `async with` awaits what `__aenter__` gives.
                enter = (
                    "__enter__()"
                    if isinstance(node, ast.With)
                    else "__aenter__().await"
                )
                for item in node.items:
                    origin = f"{self._origin(item.context_expr, lineno)}.{enter}"
                    bind(item.optional_vars, _py_start(node.body[0]), origin)
            elif isinstance(node, ast.Lambda):
                for arg in ast.walk(node.args):
                    if isinstance(arg, ast.arg):
                        self._bind_scoped(node.body, ast.Name(arg.arg), "?")
            elif isinstance(node, (ast.If, ast.IfExp, ast.While)):
                held = node.body if isinstance(node.body, list) else [node.body]
                for name, origin in checked(node.test):
                    narrow([node.test, *held], name, origin)
            elif isinstance(node, ast.Assert):
                for name, origin in checked(node.test):
                    if name.id in frame and "?" not in origin:
                        bind(name, _py_end(node), origin)
            elif isinstance(node, ast.Match) and isinstance(node.subject, ast.Name):
                for case in node.cases:
                    pattern = getattr(case.pattern, "pattern", None) or case.pattern
                    if isinstance(pattern, ast.MatchClass):
                        origin = self._instance_origin(pattern.cls)
                        narrow(case.body, node.subject, origin)
            elif isinstance(node, _PY_NAME_BINDERS) and node.name:
                origin = "?"
                if isinstance(getattr(node, "pattern", None), ast.MatchClass):
                    origin = self._instance_origin(node.pattern.cls)
                held = getattr(node, "body", None)
                at = _py_start(held[0]) if held else _py_end(node)
                put(node.name, at, origin)
            elif isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)):
                self.exits.append(node)
        # Past a statement that runs a body only where its head allows, a
        # name holds what each way through it that goes on past it and has
        # the name bound gives, where they agree, else no known value. Inner
        # statements go first.
        # Line -> the names ``frame`` binds on it, and its keys sorted: only
        # a name bound inside a branch can disagree through it.
        frame_lines = {}
        for name, bound in frame.items():
            for at, _ in bound:
                frame_lines.setdefault(at[0], set()).add(name)
        frame_order = sorted(frame_lines)
        for node, first, ways, spans in sorted(
            branching, key=lambda b: (_py_end(b[0]), -b[1][0])
        ):
            leaves = (ast.Return, ast.Raise)
            if not isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                leaves += (ast.Continue, ast.Break)
            on = [
                s for w, s in zip(ways, spans) if not (w and isinstance(w[-1], leaves))
            ]
            for name in self._bound_names(
                frame_lines, frame_order, first[0], _py_end(node)[0]
            ):
                # What the statement binds, and the last bound before it
                bound = frame[name]
                part = bound[
                    max(
                        bisect_left(bound, first, key=itemgetter(0)) - 1, 0
                    ) : bisect_right(bound, _py_end(node), key=itemgetter(0))
                ]
                if _disagree(part, first, on):
                    put(name, _py_end(node), "?")
        if func is None:
            written, listed = _py_all(body)
            if written:
                self.builder.py_all[self.builder.current_file_id] = listed
            stars = module_bound.pop("*", [])
            for name, bound in module_bound.items():
                # What a function reads is what the module bound last: the
                # last binding, and those before it back to one no
                # conditional holds. A star import after a definition is
                # one of its bindings, which the resolver tells by what the
                # module it names exports.
                if any(kind == "def" for kind, *_ in bound):
                    bound = sorted(bound + stars, key=itemgetter(2))
                last = set()
                for kind, conditional, _ in reversed(bound):
                    last.add(kind)
                    if not conditional:
                        break
                if last == {"import", "def"}:
                    self.import_binding[name] = "either"
                elif last == {"import"} and any(b[:2] == ("def", False) for b in bound):
                    self.import_binding[name] = "import"

    def _class_attrs(self, class_node, class_id):
        """Map each attribute of ``class_node`` to the origin of the value
        it is annotated with or of what a property of that name is
        annotated to return, else to the origins of the values other than
        `None` that its body and its methods assign to it. Record those its
        methods assign to the instance, `self.name = ...`, but for a
        property's, which its setter takes, as ``class_id``'s."""
        attrs = {}
        assigned = {}
        instance, properties = {}, set()
        for stmt in class_node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                origin = self._instance_origin(stmt.annotation)
                attrs.setdefault(stmt.target.id, origin)
            elif isinstance(stmt, ast.Assign) and not _py_is_none(stmt.value):
                origin = self._origin(stmt.value, stmt.lineno)
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        assigned.setdefault(target.id, set()).add(origin)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                made = [self._origin(d, stmt.lineno) for d in stmt.decorator_list]
                read = any(map(_py_attribute, made))
                if read:
                    properties.add(stmt.name)
                if read and stmt.returns:
                    origin = self._instance_origin(stmt.returns)
                    attrs.setdefault(stmt.name, origin)
                self._push_bindings(stmt.body, stmt, is_method=True)
                returned, yields = self._returned_origins()
                if read and not stmt.returns and not yields:
                    assigned.setdefault(stmt.name, set()).update(returned)
                for node in ast.walk(stmt):
                    if isinstance(node, ast.Assign):
                        targets, value = node.targets, node.value
                    elif isinstance(node, ast.AnnAssign):
                        targets, value = [node.target], node.annotation
                    else:
                        continue
                    for target in targets:
                        if (
                            not isinstance(target, ast.Attribute)
                            or self._origin(target.value, node.lineno) != "self"
                        ):
                            continue
                        held = instance.setdefault(target.attr, set())
                        if isinstance(node, ast.AnnAssign):
                            origin = self._instance_origin(value)
                            attrs.setdefault(target.attr, origin)
                            held.add(origin)
                        elif not _py_is_none(value):
                            origin = self._origin(value, node.lineno)
                            assigned.setdefault(target.attr, set()).add(origin)
                            held.add(origin)
                self.bindings.pop()
        self.builder.assigned[class_id] = {
            name: sorted(origins)
            for name, origins in instance.items()
            if name not in properties
        }
        held = {name: sorted(origins) for name, origins in assigned.items()}
        held.update((name, [origin]) for name, origin in attrs.items())
        return {
            name: origins
            for name, origins in held.items()
            if not any("?" in origin for origin in origins)
        }

    def _get_attribute_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_attribute_name(node.value)
            return f"{value}.{node.attr}" if value else node.attr
        return None
