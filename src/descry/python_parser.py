"""The Python parser."""

import ast
import re
from collections import deque
from itertools import count
from operator import itemgetter

from descry import _stdlib
from descry._parsing import (
    BaseParser,
    SymbolTable,
    _bracketed_parts,
    _code_units,
    _scip_columns,
    logger,
    merge_import,
    source_lines,
    token_count,
)

_PY_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_PY_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
# The values a `%N` holds as its own before it holds the keys as they are
_PY_HELD = 16
# The steps an origin is written with before a `%N` of its own stands for it
_PY_STEPS = 16


def _py_start(node):
    """The line and column ``node`` starts at, its decorators included."""
    first = (getattr(node, "decorator_list", None) or [node])[0]
    return first.lineno, first.col_offset


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


_PY_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


# The origins of the constant that is true only for a type checker
_PY_TYPE_CHECKING = ("@typing.TYPE_CHECKING", "@typing_extensions.TYPE_CHECKING")

# The methods Python passes the class without a decorator saying so
_PY_CLASS_TAKING = ("__new__", "__init_subclass__", "__class_getitem__")

# The key, no name, under which a state holds that only a type checker
# reads what runs in it
_PY_CHECKER = ""


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


def _py_simple_statements(statements):
    """Yield the statements among ``statements`` that hold no block of
    them, and those in the blocks of the others, in the order written."""
    todo = deque(statements)
    while todo:
        stmt = todo.popleft()
        if isinstance(stmt, _PY_COMPOUND_STMTS):
            todo.extendleft(reversed(list(_py_block_statements(stmt))))
        else:
            yield stmt


def _py_nested_definitions(statements):
    """Yield the definitions among ``statements``, looking through compound
    statements but not into other definitions."""
    for stmt in _py_simple_statements(statements):
        if isinstance(stmt, _PY_DEFINITIONS):
            yield stmt


def _py_bound_names(statements):
    """Yield the names ``statements`` bind, looking through compound
    statements but not into definitions."""
    for stmt in _py_simple_statements(statements):
        if isinstance(stmt, _PY_DEFINITIONS):
            yield stmt.name
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            yield from (name for name, _ in _py_import_bindings(stmt))
        else:
            yield from (
                node.id
                for node in ast.walk(stmt)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            )


class _PyFrame(dict):
    """A frame of ``PythonParser.bindings``: each name -> the origins it is
    bound to; ``inner``, the names the body binds, and ``stored``, those it
    and the parameters bind to a value, not by `def`, `class` or `import`."""


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
        self.function_ids = set()
        self.import_targets = {}
        # name -> "import" where the module's import of it rebinds its
        # definition, "either" where which one binds is not told
        self.import_binding = {}
        self.bindings = []
        # id() of a name that is read -> origin of what it holds there
        self.scoped = {}
        # (`%` or `&`, the origins a key of ``builder.literals`` holds) -> the key
        self.arms = {}
        # `&N` -> the value it was tested on
        self.tested = {}
        # id() of each definition that follows, in the block it is a
        # statement of, another of its name
        self.redefined = set()
        # id() of each definition a class body holds -> the names the body binds
        self.class_bound = {
            id(definition): names
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            for names in [set(_py_bound_names(node.body))]
            for definition in _py_nested_definitions(node.body)
        }
        # The names the file declares `global`
        self.rebound = {
            name
            for node in ast.walk(tree)
            if isinstance(node, ast.Global)
            for name in node.names
        }
        # Whether a star import may bind any name of the file
        self.starred = any(
            alias.name == "*"
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        )
        self._push_bindings(tree.body)
        # A name bound only as `global`, by a function or after a star
        # import, has no known value.
        self.builder.fields[file_id] = {name: ["?"] for name in self.rebound} | {
            name: sorted(bound) for name, bound in self.bindings[0].items()
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
                origin = self._origin(base, node.lineno)
                base_name = self._get_attribute_name(base)
                if not base_name and origin.endswith(".class"):
                    # `type(x)`, named as written
                    base_name = ast.unparse(base)
                if base_name:
                    resolved_base = self.builder.current_scope.resolve(base_name)
                    self.builder.add_edge(
                        class_id,
                        f"REF:{resolved_base}",
                        "INHERITS",
                        {
                            "origin": origin,
                            # The base's own position: SCIP resolves the
                            # edge at its occurrence, not by bare name.
                            "lineno": base.lineno,
                            "col": base.col_offset,
                        },
                    )

            for keyword in node.keywords:
                if keyword.arg == "metaclass":
                    origin = self._origin(keyword.value, node.lineno)
                    self.builder.py_metaclass[class_id] = origin
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

            again = func_id in self.function_ids and id(node) not in self.redefined
            self.function_ids.add(func_id)
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
            metadata = self.builder.nodes_by_id[func_id]["metadata"]
            self.builder.add_edge(parent_id, func_id, "DEFINES")

            self._record_decorators(node, func_id)
            if node.decorator_list and not is_method:
                # The name is bound to what the outermost decorator returns.
                made = self._origin(node.decorator_list[0], node.lineno)
                bound = self.builder.fields.setdefault(parent_id, {})
                bound[node.name] = sorted(
                    f"{made}()" if origin == node.name else origin
                    for origin in bound.get(node.name, [node.name])
                )

            self._push_bindings(node.body, node, is_method)
            returns = self._returns(node)
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
            plain = not (returns or yields or made - {"staticmethod", "classmethod"})
            # An async generator's item is what its awaited `__anext__` gives.
            step = (
                "__anext__().await"
                if isinstance(node, ast.AsyncFunctionDef)
                else "__next__()"
            )
            if manager and returns:
                origins = [f"{self._instance_origin(returns)}.{step}"]
            held = self.builder.fields.setdefault(func_id, {})
            key = manager or "()"
            if again:
                # A second definition under the name, on another way: what
                # either gives
                origins = key in held and origins and sorted({*origins, *held[key]})
                held.pop(key, None)
            if origins and (plain or manager):
                held[key] = origins
            elif node.returns and not returns:
                # Neither what it returns nor its annotation is known.
                held["()"] = ["?"]
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
            for child in _py_simple_statements([node]):
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
        origin = self._origin(func_node, lineno)
        if func_name or "?" not in origin:
            # What an expression that is no name gives is called as written.
            resolved_name = (
                self.builder.current_scope.resolve(func_name)
                if func_name
                else ast.unparse(func_node)
            )
            if isinstance(func_node, ast.Name):
                # A variable that is called holds a value, `=name`, which
                # returns `Type` where it is annotated as a callable that
                # does: `=name>Type`.
                annotated = re.fullmatch(r":(?:\w+::)*Callable\[(.+)\]", origin)
                if origin == "?" and func_name not in self.bindings[-1].get(
                    func_name, ()
                ):
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
            for frame in reversed(self.bindings):
                if node.id in frame:
                    return frame[node.id][-1]
            imported = self.import_targets.get(node.id)
            return f"@{imported}" if isinstance(imported, str) else node.id
        if isinstance(node, ast.Attribute):
            return f"{self._origin(node.value, lineno)}.{node.attr}"
        if isinstance(node, ast.Call):
            made = self._origin(node.func, lineno)
            first = self._origin(node.args[0], lineno) if node.args else "?"
            if "type" not in self.rebound and not any(
                "type" in frame for frame in self.bindings
            ):
                if made == "type" and len(node.args) == 1 and not node.keywords:
                    # The builtin's `type(x)` is the class of `x`.
                    return f"{first}.class"
                if made == "type.__call__" and first not in ("self", "cls"):
                    return "?"
            if node.args and made.endswith(".__new__"):
                # `__new__` makes an instance of the class it is passed.
                made = f"{first}.__new__"
            return f"{made}()"
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
        for node in self.exits.values():
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
        """Push a frame mapping each name ``body`` (and the parameters of
        ``func``) binds to the origins it is bound to, the last of them what
        it holds where the body ends, and record in ``scoped`` what each
        name ``body`` reads holds where it is read. The statements are
        followed as they run, in ``env``: each name -> ``(origin,
        narrowed)``, what it is bound to and what a test passed on the way
        tells of it, either None. A way through a statement is run, kept as
        the names it changed, and undone; the ways that go on are joined
        past the statement. Module-level imports go to ``import_targets``."""
        frame = _PyFrame()
        self.bindings.append(frame)
        self.exits = {}  # id() -> the `return` and `yield` nodes of ``body``
        params = []
        if func is not None:
            args = func.args
            params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
            params += [a for a in (args.vararg, args.kwarg) if a]
        inner, outer = set(), set()
        stored = {arg.arg for arg in params}
        todo = deque(body)
        while todo:
            node = todo.popleft()
            if isinstance(node, (ast.Lambda, *_PY_COMPREHENSIONS)) or (
                isinstance(node, ast.Assign) and _py_is_none(node.value)
            ):
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if func is not None:
                    inner.update(name for name, _ in _py_import_bindings(node))
                continue
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                outer.update(node.names)
            if isinstance(node, (ast.Yield, ast.YieldFrom)):
                # A `yield` makes a generator of the function wherever it is.
                self.exits[id(node)] = node
            names = {getattr(node, "name", None), getattr(node, "rest", None)}
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            names.discard(None)
            inner |= names
            if not isinstance(node, _PY_DEFINITIONS):
                stored |= names
                todo.extend(ast.iter_child_nodes(node))
        inner -= outer
        frame.inner, frame.stored = inner, stored
        # The names the frame binds: one of them read where no way to the
        # read has bound it holds nothing known.
        local = inner | {arg.arg for arg in params}
        # name -> ("def" or "import", whether a conditional holds it, where
        # it is written) of each binding of it at module level; a star
        # import, which may bind any name, is filed under `*`
        module_bound = {}
        env, log = {}, []  # ``log``: each change of ``env`` as (name, value before)
        nothing = None, None
        sinks = []  # name -> the origins bound in it, per `try` being run
        loops = []  # (where in ``log`` it began, the changes at each `continue`, at each `break`), per loop being run
        conditional = 0  # the statements being run that may skip a block
        apart = 0  # the class bodies, comprehensions and lambdas being read
        lambdas = 0  # the lambdas being read
        watched = {}  # name -> ID of a read of it that runs later -> (the read, the origins it may hold then)
        # The names bound elsewhere too, which hold no known value: those
        # the frame or a function nested in it declares `global` or
        # `nonlocal`, and a module's that a function declares `global`
        loose = outer if func else set(self.rebound)

        def known(origin):
            """``origin``, or `?` where it is no key and has no known value."""
            return (
                "?" if "?" in origin and origin not in self.builder.literals else origin
            )

        def either(origins):
            """The origin of a name that holds one of ``origins``: `%N`, the
            entry of ``builder.literals`` that holds each, where the
            resolver finds what they agree on; `%?N` where one of them is
            not known, which reads as `?` does. A `%N` among them gives its
            own, unless that makes more than ``_PY_HELD``; one origin is
            its own."""
            literals = self.builder.literals
            origins = {known(origin) for origin in origins}
            if len(origins) == 1:
                return origins.pop()
            held = {
                known(arm)
                for origin in origins
                for arm in (
                    literals[origin]
                    if origin[0] == "%" and origin in literals
                    else [origin]
                )
            }
            if len(held) > _PY_HELD:
                held = origins
            if len(held) == 1:
                return held.pop()
            unknown = any("?" in arm for arm in held)
            return kept("%?" if unknown else "%", tuple(sorted(held)))

        def kept(sigil, origins):
            """The key of ``builder.literals`` that holds ``origins``."""
            if (sigil, origins) not in self.arms:
                key = f"{sigil}{self.builder.serial(self.builder.literals)}"
                self.builder.literals[key] = list(origins)
                self.arms[sigil, origins] = key
            return self.arms[sigil, origins]

        def apply(changes):
            for name, value in changes.items():
                log.append((name, env.get(name, nothing)))
                env[name] = value

        def undo(mark):
            while len(log) > mark:
                name, env[name] = log.pop()

        def changed(mark):
            """What ``env`` holds of each name changed since ``mark``."""
            return {name: env[name] for name, _ in log[mark:]}

        def passed(origin, value):
            """Whether ``origin`` is ``value`` as tests of it passed it."""
            while origin in self.tested:
                origin = self.tested[origin]
                if origin == value:
                    return True
            return False

        def joined(ways):
            """The changes of ``env`` that give what holds where any of
            ``ways``, each the changes of one or None, may have run: a name
            they bind differently is bound to what each that binds it tells
            or holds of it, and one they bind alike is told what each
            tells or holds. What only a type checker reads never runs: it
            gives only the names nothing else binds. None where there is
            no way."""
            ways = [way for way in ways if way is not None]
            seen = [
                any(way.get(_PY_CHECKER, env.get(_PY_CHECKER, nothing))) for way in ways
            ]
            if any(seen) and not all(seen):
                runs = joined(way for way, checker in zip(ways, seen) if not checker)
                for way in ways:
                    for name, value in way.items():
                        if name not in runs and not any(env.get(name, nothing)):
                            runs[name] = value
                return {**runs, _PY_CHECKER: nothing}
            changes = {}
            for name in dict.fromkeys(name for way in ways for name in way):
                values = {way.get(name, env.get(name, nothing)) for way in ways}
                value = next(iter(values))
                if len(values) > 1:
                    origins = {told or origin for origin, told in values}
                    if name in origins:
                        # A definition of the name and an import of it:
                        # ``import_binding`` tells which binds.
                        origins -= {told for origin, told in values if not origin}
                    here = env.get(name, nothing)
                    before = here[1] or here[0]
                    if not all(name in way for way in ways):
                        # What a way made of the value a test of it passed
                        # joins the value, where another way leaves it.
                        origins -= {
                            origin for origin in origins if passed(origin, before)
                        }
                    if len({origin for origin, _ in values}) > 1:
                        value = either(origins - {None}), None
                    elif all(
                        origin == before or passed(origin, before) for origin in origins
                    ):
                        # Each way holds what was held before.
                        value = here
                    elif value[0] is None:
                        # A name bound outside the function, or a module's
                        # import, is told nothing.
                        value = None, None
                    else:
                        value = value[0], either(origins)
                changes[name] = value
            return changes if ways else None

        def merge(ways):
            """Whether any of ``ways`` goes on, ``env`` then what they join to."""
            changes = joined(ways)
            apply(changes or {})
            return changes is not None

        def way(mark, changes, statements):
            """What ``statements``, run where ``changes`` are made, leave
            changed since ``mark``, None where they do not go on."""
            here = len(log)
            apply(changes)
            left = changed(mark) if run(statements) else None
            undo(here)
            return left

        def put(name, origin):
            if not apart:
                if name in loose:
                    origin = "?"
                elif origin.count(".") > _PY_STEPS and "?" not in origin:
                    origin = kept("%", (origin,))
                bound = frame.setdefault(name, [])
                if origin not in bound:
                    bound.append(origin)
                for sink in sinks:
                    sink.setdefault(name, set()).add(origin)
                for _, seen in watched.get(name, {}).values():
                    seen.add(origin)
            apply({name: (origin, None)})

        def watch(node):
            """Record the read ``node``, which runs when what holds it is
            called: its name may hold then what it does here or is bound
            to from here on."""
            if func and node.id in local:
                origin, narrowed = env.get(node.id, nothing)
                reads = watched.setdefault(node.id, {})
                reads.setdefault(id(node), (node, set()))[1].add(narrowed or origin)

        def later(node):
            """Watch the names read in the functions of the definition
            ``node``, and leave open those they bind as `nonlocal`."""
            todo = [(node, False)]
            while todo:
                part, inside = todo.pop()
                inside = inside or isinstance(part, _PY_FUNCTIONS)
                if isinstance(part, ast.Nonlocal):
                    for name in local.intersection(part.names):
                        loose.add(name)
                        put(name, "?")
                elif inside and isinstance(part, ast.Name):
                    watch(part)
                todo.extend((child, inside) for child in ast.iter_child_nodes(part))

        def maybe(changes):
            """Bind what ``changes`` bind, where what made them may not
            have run."""
            here = len(log)
            for name, (origin, _) in changes.items():
                put(name, origin)
            bound = changed(here)
            undo(here)
            merge([{}, bound])

        def bind(target, origin):
            for name, bound in _py_targets(target, origin):
                put(name, bound)

        def partly(sink):
            """The changes of ``env`` where a block that bound ``sink`` ran
            in part."""
            return {
                name: (
                    either([*bound, *filter(None, env.get(name, nothing)[:1])]),
                    None,
                )
                for name, bound in sink.items()
            }

        def iterated(value, loop):
            """Origin of what ``loop`` binds on each pass over ``value``."""
            step = "__iter__().__next__()"
            if isinstance(loop, ast.AsyncFor) or getattr(loop, "is_async", 0):
                step = "__aiter__().__anext__().await"
            return f"{self._origin(value, value.lineno)}.{step}"

        def valued(node, lineno):
            """Origin of the value of the expression ``node``: either arm's
            of a conditional expression, any operand's of `and` and `or`."""
            if isinstance(node, (ast.IfExp, ast.BoolOp)):
                parts = getattr(node, "values", None) or (node.body, node.orelse)
                return either(valued(part, lineno) for part in parts)
            return self._origin(node, lineno)

        def read(node):
            """Evaluate the expressions of ``node``: record what each name
            they read holds, and bind what they bind."""
            nonlocal apart, lambdas
            todo = [node]
            while todo:
                node = todo.pop()
                if isinstance(node, ast.Name):
                    origin, narrowed = env.get(node.id, nothing)
                    if not isinstance(node.ctx, ast.Load):
                        continue
                    if origin or narrowed:
                        self.scoped[id(node)] = narrowed or origin
                    elif node.id in local:
                        self.scoped[id(node)] = "?"
                    if lambdas:
                        watch(node)
                elif isinstance(node, (ast.BoolOp, ast.IfExp)) or (
                    len(getattr(node, "comparators", ())) > 1
                ):
                    merge(cond(node))
                elif isinstance(node, ast.NamedExpr):
                    read(node.value)
                    bind(node.target, valued(node.value, node.lineno))
                elif isinstance(node, ast.Lambda):
                    # Its body runs when it is called, and binds its own names.
                    read(node.args)
                    here = len(log)
                    apart += 1
                    apply(
                        {
                            arg.arg: ("?", None)
                            for arg in ast.walk(node.args)
                            if isinstance(arg, ast.arg)
                        }
                    )
                    lambdas += 1
                    read(node.body)
                    lambdas -= 1
                    apart -= 1
                    undo(here)
                elif isinstance(node, _PY_COMPREHENSIONS):
                    here = len(log)
                    apart += 1
                    targets = set()
                    # A generator expression runs when it is iterated, but
                    # for its first iterable.
                    waits = isinstance(node, ast.GeneratorExp)
                    for at, generator in enumerate(node.generators):
                        read(generator.iter)
                        lambdas += waits and not at
                        origin = iterated(generator.iter, generator)
                        bind(generator.target, origin)
                        targets.update(n for n, _ in _py_targets(generator.target, ""))
                        for test in generator.ifs:
                            apply(cond(test)[0])
                    for part in ast.iter_child_nodes(node):
                        if not isinstance(part, ast.comprehension):
                            read(part)
                    apart -= 1
                    lambdas -= waits
                    bound = changed(here)
                    undo(here)
                    # A `:=` in it binds the name for what holds it, where
                    # a pass ran.
                    maybe(
                        {
                            name: (origin, None)
                            for name, (origin, _) in bound.items()
                            if name not in targets
                            and origin != env.get(name, nothing)[0]
                        }
                    )
                else:
                    parts = list(ast.iter_child_nodes(node))
                    if isinstance(node, ast.Dict):
                        # Each key is evaluated before its value.
                        pairs = zip(node.keys, node.values)
                        parts = [part for pair in pairs for part in pair if part]
                    elif isinstance(node, ast.arguments):
                        # Defaults are evaluated before those of the
                        # keyword-only parameters.
                        rest = [part for part in parts if part not in node.defaults]
                        parts = [*node.defaults, *rest]
                    elif isinstance(node, ast.AnnAssign):
                        # The value is evaluated first, a local's annotation
                        # not at all.
                        parts = [node.value, node.target]
                        parts += [] if func else [node.annotation]
                        parts = list(filter(None, parts))
                    todo.extend(
                        part
                        for part in reversed(parts)
                        if not isinstance(part, ast.stmt)
                    )

        def tested(node, written):
            """The change of ``env`` where the name ``node`` reads is told
            to be of the class ``written``: `&N`, each value it may hold
            where that is of the class, else an instance of the class. A
            name bound elsewhere too is told nothing."""
            told = self._instance_origin(written)
            name = node.id
            origin, narrowed = env.get(name, nothing)
            if not re.fullmatch(r":[\w:]+", told) or (
                name in loose | self.rebound - local
            ):
                return {}
            if self._bound_to_value(told[1:].split("::")[0], self.bindings[1:]):
                return {}
            held = narrowed or origin or name not in local and self._origin(node, 0)
            held = known(held or "?")
            if held not in self.tested or self.builder.literals[held][1] != told:
                held, on = kept("&", (held, told)), held
                self.tested[held] = on
            return {name: (origin, held)}

        def cond(test):
            """``(held, other)``: the changes of ``env`` past ``test``,
            where it holds and where it does not. ``env`` is left as it
            was."""
            if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
                return cond(test.operand)[::-1]
            mark = len(log)
            if isinstance(test, ast.IfExp):
                ends = []
                for changes, part in zip(cond(test.test), (test.body, test.orelse)):
                    apply(changes)
                    ends.append([{**changed(mark), **end} for end in cond(part)])
                    undo(mark)
                return tuple(map(joined, zip(*ends)))
            chained = len(getattr(test, "comparators", ())) > 1
            if isinstance(test, ast.BoolOp) or chained:
                # An operand is evaluated where those before it did not
                # settle the result; `a < b < c` is `a < b and b < c`.
                parts = (
                    [ast.Tuple([test.left, test.comparators[0]]), *test.comparators[1:]]
                    if chained
                    else test.values
                )
                order = -1 if isinstance(getattr(test, "op", None), ast.Or) else 1
                settled = []
                for part in parts:
                    held, other = cond(part)[::order]
                    settled.append({**changed(mark), **other})
                    apply(held)
                held = changed(mark)
                undo(mark)
                return (held, joined(settled))[::order]
            read(test)
            held = other = changed(mark)
            if (
                "TYPE_CHECKING" in (getattr(test, "id", ""), getattr(test, "attr", ""))
                and self._origin(test, test.lineno) in _PY_TYPE_CHECKING
            ):
                held = {**other, _PY_CHECKER: (None, True)}
            elif (
                isinstance(test, ast.Call)
                and isinstance(test.func, ast.Name)
                and test.func.id == "isinstance"
                and len(test.args) == 2
                and isinstance(test.args[0], ast.Name)
            ):
                held = {**other, **tested(test.args[0], test.args[1])}
            undo(mark)
            return held, other

        def loop(node):
            """Whether the loop ``node`` goes on. Its body is run until
            what holds where a pass starts is what the passes before it
            leave."""
            origin = None
            if not isinstance(node, ast.While):
                read(node.iter)
                origin = iterated(node.iter, node)
            mark = len(log)
            head, unknown = {}, {}
            sink = {}
            sinks.append(sink)
            for attempt in count():
                apply(head)
                done = head
                if origin:
                    bind(node.target, origin)
                else:
                    held, other = cond(node.test)
                    done = {**head, **other}
                    apply(held)
                loops.append((mark, [], []))
                end = changed(mark) if run(node.body) else None
                _, again, out = loops.pop()
                undo(mark)
                after = {**joined([{}, end, *again]), **unknown}
                if after == head:
                    break
                if attempt > 1:
                    # A name each pass binds to something new holds what
                    # it did before the loop or a value not known.
                    for name, value in after.items():
                        if value != head.get(name):
                            was, told = env.get(name, nothing)
                            unknown[name] = either(["?", told or was or "?"]), None
                    after = {**after, **unknown}
                head = after
            sinks.pop()
            # A read that runs later may run on a later pass, past what any
            # part of the body binds.
            for name, bound in sink.items():
                for _, seen in watched.get(name, {}).values():
                    seen |= bound
            return merge([way(mark, done, node.orelse), *out])

        def run(statements):
            """Whether ``statements``, run in ``env``, go on past their
            last."""
            nonlocal conditional, apart
            defined = set()
            for node in statements:
                lineno = node.lineno
                mark = len(log)
                skips = isinstance(
                    node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Match)
                ) or bool(getattr(node, "handlers", ()))
                conditional += skips
                goes = True
                if isinstance(node, ast.If):
                    ways = []
                    while True:
                        held, other = cond(node.test)
                        ways.append(way(mark, held, node.body))
                        apply(other)
                        if [type(s) for s in node.orelse] != [ast.If]:
                            break
                        node = node.orelse[0]
                    ways.append(way(mark, {}, node.orelse))
                    undo(mark)
                    goes = merge(ways)
                elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                    goes = loop(node)
                elif isinstance(node, (ast.Try, ast.TryStar)):
                    sink = {}
                    sinks.append(sink)
                    jumps = (loops or [()])[-1][1:]
                    before = [len(left) for left in jumps]
                    end = changed(mark) if run(node.body) else None
                    undo(mark)
                    raised = partly(sink)
                    ways = [None if end is None else way(mark, end, node.orelse)]
                    ways += [way(mark, raised, [handler]) for handler in node.handlers]
                    sinks.pop()
                    # `finally` runs whatever ran before it, a `continue`
                    # or a `break` too, and what it binds holds past it.
                    apply(partly(sink))
                    here = len(log)
                    through = [left[at:] for left, at in zip(jumps, before)]
                    ended = run(node.finalbody)
                    bound = changed(here)
                    undo(mark)
                    for left, at, crossing in zip(jumps, before, through):
                        left[at : at + len(crossing)] = [
                            {**changes, **bound} for changes in crossing if ended
                        ]
                    goes = merge(ways) and ended
                    apply(bound)
                elif isinstance(node, ast.ExceptHandler):
                    read(node)
                    if node.name:
                        put(node.name, "?")
                    goes = run(node.body)
                elif isinstance(node, ast.Match):
                    read(node.subject)
                    mark = len(log)
                    ways = []
                    for case in node.cases:
                        missed = changed(mark)
                        read(case.pattern)
                        captured = {}
                        for part in ast.walk(case.pattern):
                            name = getattr(part, "name", getattr(part, "rest", None))
                            if name:
                                origin = "?"
                                if isinstance(
                                    getattr(part, "pattern", 0), ast.MatchClass
                                ):
                                    origin = self._instance_origin(part.pattern.cls)
                                captured[name] = origin
                        pattern = getattr(case.pattern, "pattern", None) or case.pattern
                        if isinstance(pattern, ast.MatchClass) and isinstance(
                            node.subject, ast.Name
                        ):
                            apply(tested(node.subject, pattern.cls))
                            told = env.get(node.subject.id, nothing)[1]
                            if told and getattr(case.pattern, "name", None):
                                captured[case.pattern.name] = told
                        for capture in captured.items():
                            put(*capture)
                        other = None
                        if case.guard:
                            # A guard that fails leaves what the pattern
                            # and it bound to the cases after it.
                            held, other = cond(case.guard)
                            other = {**changed(mark), **other}
                            apply(held)
                        elif isinstance(pattern, ast.MatchAs):
                            missed = None  # it matches anything
                        ways.append(way(mark, {}, case.body))
                        undo(mark)
                        if missed is None:
                            break
                        apply(joined([missed, other]))
                    else:
                        ways.append(changed(mark))
                        undo(mark)
                    goes = merge(ways)
                elif isinstance(node, (ast.With, ast.AsyncWith)):
                    # `async with` awaits what `__aenter__` gives.
                    enter = (
                        "__enter__()"
                        if isinstance(node, ast.With)
                        else "__aenter__().await"
                    )
                    for item in node.items:
                        read(item)
                        origin = f"{self._origin(item.context_expr, lineno)}.{enter}"
                        bind(item.optional_vars, origin)
                    here = len(log)
                    sink = {}
                    sinks.append(sink)
                    end = changed(here) if run(node.body) else None
                    sinks.pop()
                    undo(here)
                    # What is entered may take what the body raises: the
                    # statements after it run past a body run in part.
                    goes = merge([end, partly(sink)])
                elif isinstance(node, ast.Assert):
                    held, other = cond(node.test)
                    if node.msg:
                        apply(other)
                        read(node.msg)
                        undo(mark)
                    apply(held)
                elif isinstance(node, ast.Assign):
                    read(node.value)
                    for target in node.targets:
                        read(target)
                    if not _py_is_none(node.value):
                        origin = valued(node.value, lineno)
                        for target in node.targets:
                            bind(target, origin)
                elif isinstance(node, ast.AugAssign):
                    origin, narrowed = env.get(getattr(node.target, "id", ""), nothing)
                    read(node)
                    if isinstance(node.target, ast.Name):
                        # `=__add__`: the in-place method where what the
                        # name holds has one, else the operator's. A library
                        # type stays where that method gives its own type.
                        held = narrowed or origin or "?"
                        method = _PY_OPERATORS[type(node.op)]
                        made = held.removeprefix("@builtins.")[:-2]
                        if self.starred or any(made in f for f in self.bindings):
                            made = ""
                        methods = (
                            _stdlib.load("python")["types"]
                            .get(f"builtins.{made}", {})
                            .get("methods", {})
                        )
                        gives = methods.get(f"__i{method[2:]}") or methods.get(method)
                        if not gives or gives.partition("[")[0] not in (
                            "typing.Self",
                            made,
                        ):
                            put(node.target.id, f"{held}.={method}()")
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for name, target in _py_import_bindings(node):
                        if func is not None or apart or name in loose:
                            put(name, f"@{target}")
                            continue
                        entry = ("import", conditional > 0, _py_start(node))
                        bound = module_bound.setdefault(
                            "*" if target.endswith(".*") else name, []
                        )
                        if entry not in bound:
                            bound.append(entry)
                        if conditional:
                            # An import a conditional holds and another of
                            # its ways may bind: which one is not told. It
                            # merges with the targets already there.
                            merge_import(self.import_targets, name, target)
                        else:
                            # An unconditional import rebinds the name:
                            # the later one is what binds after it.
                            self.import_targets[name] = target
                        # What the name is told to be on this way, for
                        # where another assigns it.
                        apply({name: (None, f"@{target}")})
                else:
                    read(node)
                    if isinstance(node, _PY_DEFINITIONS):
                        if isinstance(node, ast.ClassDef):
                            apart += 1
                            run(node.body)
                            apart -= 1
                            undo(mark)
                        if not apart:
                            later(node)
                        if node.name in defined:
                            self.redefined.add(id(node))
                        defined.add(node.name)
                        if not apart:
                            # A `def` or `class` binds its name for what is
                            # written after it, which shadows an import of
                            # the name.
                            frame.pop(node.name, None)
                            if func is None:
                                entry = ("def", conditional > 0, _py_start(node))
                                bound = module_bound.setdefault(node.name, [])
                                if entry not in bound:
                                    bound.append(entry)
                        put(node.name, node.name)
                    elif isinstance(node, ast.AnnAssign):
                        bind(node.target, self._instance_origin(node.annotation))
                    elif isinstance(node, (ast.Break, ast.Continue)):
                        begun, *left = loops[-1]
                        left[isinstance(node, ast.Break)].append(changed(begun))
                    if isinstance(node, ast.Return):
                        self.exits[id(node)] = node
                    goes = not isinstance(
                        node, (ast.Return, ast.Raise, ast.Break, ast.Continue)
                    )
                conditional -= skips
                if not goes:
                    return False
            return True

        if func is not None:
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
                    and ("staticmethod" not in decorators or func.name == "__new__")
                ):
                    held = "classmethod" in decorators or func.name in _PY_CLASS_TAKING
                    origin = "cls" if held else "self"
                elif arg is args.vararg:
                    origin = "@builtins.tuple()"
                elif arg is args.kwarg:
                    origin = "@builtins.dict()"
                elif arg.annotation:
                    origin = self._instance_origin(arg.annotation)
                    if self._unreadable(origin, self.class_bound.get(id(func), set())):
                        origin = "?"
                elif injected and arg.arg not in parametrized:
                    origin = f"~{arg.arg}"
                else:
                    origin = "?"
                put(arg.arg, origin)
        run(body)
        for reads in watched.values():
            for node, seen in reads.values():
                seen.discard(None)
                if seen:
                    self.scoped[id(node)] = either(seen) if len(seen) > 1 else min(seen)
        for name, bound in frame.items():
            origin = env[name][1] or env[name][0]
            if origin:
                if origin in bound:
                    bound.remove(origin)
                bound.append(origin)
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
        property's, which its setter takes, as ``class_id``'s, and for each
        origin a method wrote, the methods it is read in, where Python
        evaluates it."""
        attrs = {}
        assigned = {}
        instance, properties = {}, set()
        # (attribute, origin) -> the class or method each write is read in
        scopes = {}
        for stmt in class_node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                origin = self._instance_origin(stmt.annotation)
                attrs.setdefault(stmt.target.id, origin)
            elif isinstance(stmt, ast.Assign) and not _py_is_none(stmt.value):
                origin = self._origin(stmt.value, stmt.lineno)
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        assigned.setdefault(target.id, set()).add(origin)
                        scopes.setdefault((target.id, origin), set()).add(class_id)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                made = [self._origin(d, stmt.lineno) for d in stmt.decorator_list]
                read = any(map(_py_attribute, made))
                if read:
                    properties.add(stmt.name)
                self._push_bindings(stmt.body, stmt, is_method=True)
                returns = self._returns(stmt)
                if read and returns:
                    attrs.setdefault(stmt.name, self._instance_origin(returns))
                returned, yields = self._returned_origins()
                wrote = set()
                if read and not returns and not yields:
                    assigned.setdefault(stmt.name, set()).update(returned)
                    wrote.update((stmt.name, origin) for origin in returned)
                # What a function nested in the method binds is not read.
                nested = {
                    id(inner)
                    for definition in _py_nested_definitions(stmt.body)
                    for inner in ast.walk(definition)
                }
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
                            if id(node) not in nested:
                                wrote.add((target.attr, origin))
                        elif not _py_is_none(value):
                            origin = "?"
                            if id(node) not in nested:
                                origin = self._origin(value, node.lineno)
                            assigned.setdefault(target.attr, set()).add(origin)
                            held.add(origin)
                            wrote.add((target.attr, origin))
                for name, origin in wrote:
                    scopes.setdefault((name, origin), set()).add(
                        f"{class_id}::{stmt.name}"
                    )
                self.bindings.pop()
        self.builder.py_properties[class_id] = sorted(properties)
        local = self.builder.py_local[class_id] = {}
        for (name, origin), read_in in scopes.items():
            if read_in != {class_id}:
                local.setdefault(name, {})[origin] = sorted(read_in)
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

    def _bound_to_value(self, name, frames):
        """Whether the innermost of the function ``frames`` that binds
        ``name`` binds it to a value, not by `def`, `class` or `import`."""
        for frame in reversed(frames):
            if name in frame.inner | frame.stored:
                return name in frame.stored
        return False

    def _unreadable(self, written, around=frozenset()):
        """Whether the annotation ``written`` of the function whose frame is
        the last pushed names no class descry can read: it is evaluated
        where the function is defined, and writes a name its body or
        ``around`` binds, or one the function enclosing it binds to a
        value."""
        names = set(re.findall(r"\w+", written))
        return bool(names & (self.bindings[-1].inner | around)) or any(
            self._bound_to_value(name, self.bindings[1:-1]) for name in names
        )

    def _returns(self, func):
        """The return annotation of ``func``, whose frame is the last pushed,
        or None where it is unreadable."""
        if func.returns and not self._unreadable(ast.unparse(func.returns)):
            return func.returns
        return None

    def _get_attribute_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_attribute_name(node.value)
            return f"{value}.{node.attr}" if value else node.attr
        return None
