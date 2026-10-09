"""Generate dotnet.json from the .NET shared runtime.

usage: gen_dotnet.py RUNTIME OUTPUT

RUNTIME is the directory that holds the shared runtime's assemblies
(`System.Private.CoreLib.dll` and the `System.*.dll` beside it). The
public and protected types and members of the `System` and `netstandard`
assemblies are read by reflection in PowerShell (`pwsh`), which is the
only thing of the runtime that runs.

An extension method is written as a method of the type it is written for:
``Enumerable.Where`` is a method of ``IEnumerable`1`` — which is what a
call through an instance of a type that has it as a base resolves to.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import new_type, record

DUMP = r"""
$dir = $args[0]
$out = @()
foreach ($path in Get-ChildItem $dir -Filter "*.dll") {
    $base = $path.BaseName
    if ($base -ne "netstandard" -and -not $base.StartsWith("System.")) { continue }
    $asm = $null
    try { $asm = [System.Reflection.Assembly]::LoadFrom($path.FullName) } catch {
        try { $asm = [System.Reflection.Assembly]::Load($base) } catch { continue }
    }
    $types = @()
    try { $types = $asm.GetTypes() } catch [System.Reflection.ReflectionTypeLoadException] {
        $types = $_.Exception.Types | Where-Object { $null -ne $_ }
    } catch { continue }
    foreach ($t in $types) {
        if (-not ($t.IsPublic -or $t.IsNestedPublic -or $t.IsNestedFamily -or $t.IsNestedFamORAssem)) { continue }
        $methods = @()
        foreach ($m in $t.GetMethods("Public,NonPublic,Instance,Static,DeclaredOnly")) {
            if (-not ($m.IsPublic -or $m.IsFamily -or $m.IsFamilyOrAssembly)) { continue }
            if ($m.IsSpecialName -and -not $m.Name.StartsWith("op_")) { continue }
            if ($m.IsConstructor) { continue }
            $ps = @()
            foreach ($p in $m.GetParameters()) { $ps += ($p.ParameterType.FullName ?? $p.ParameterType.ToString()) }
            $methods += @{
                name = $m.Name
                is_static = $m.IsStatic
                is_extension = ($m.IsStatic -and $ps.Count -gt 0 -and
                    $null -ne [System.Attribute]::GetCustomAttribute($m, [System.Runtime.CompilerServices.ExtensionAttribute]))
                gparams = @($m.GetGenericArguments() | ForEach-Object { $_.Name })
                ret = if ($m.ReturnType -eq [void]) { $null } else { ($m.ReturnType.FullName ?? $m.ReturnType.ToString()) }
                args = $ps
            }
        }
        $props = @()
        foreach ($p in $t.GetProperties("Public,NonPublic,Instance,Static,DeclaredOnly")) {
            $get = $p.GetGetMethod($true)
            if ($null -ne $get -and ($get.IsPublic -or $get.IsFamily -or $get.IsFamilyOrAssembly)) {
                $props += @{
                    name = $p.Name
                    is_static = $get.IsStatic
                    ret = ($p.PropertyType.FullName ?? $p.PropertyType.ToString())
                }
            }
        }
        $invoke = $null
        if ($t.IsSubclassOf([System.MulticastDelegate])) {
            $i = $t.GetMethod("Invoke")
            if ($null -ne $i) {
                $invoke = @{
                    takes = @($i.GetParameters() | ForEach-Object { $_.ParameterType.FullName ?? $_.ParameterType.ToString() })
                    ret = if ($i.ReturnType -eq [void]) { $null } else { ($i.ReturnType.FullName ?? $i.ReturnType.ToString()) }
                }
            }
        }
        $out += @{
            name = $t.FullName
            params = @($t.GetGenericArguments() | ForEach-Object { $_.Name })
            base = if ($null -ne $t.BaseType) { ($t.BaseType.FullName ?? $t.BaseType.ToString()) } else { $null }
            interfaces = @($t.GetInterfaces() | ForEach-Object { $_.FullName ?? $_.ToString() })
            methods = $methods
            props = $props
            invoke = $invoke
        }
    }
}
$out | ConvertTo-Json -Depth 7
"""

_OWNED = re.compile(r"^(?:System|netstandard)\.")


def parts(path):
    """``(name, (type arguments))`` of a full name as reflection writes it:
    ``System.Collections.Generic.List`1[T]`` gives ``List`1`` and ``T``;
    a non-generic name or a bare parameter gives it and nothing."""
    if not path or "[" not in path or "]" not in path:
        return path, ()
    name, _, rest = path.partition("[")
    inner = rest[: rest.rindex("]")]
    if inner.startswith("["):
        # An instantiated generic's arguments are written as the full,
        # assembly-qualified name of each, in brackets of their own.
        entries = re.findall(r"\[([^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*)\]", inner)
        args = [entry.split(", ")[0] for entry in entries]
    else:
        # A definition's arguments are its parameter names.
        args = [a.strip() for a in inner.split(",")]
    return name, args


_ARITY = re.compile(r"`\d+$")


def written(path, names):
    """The type ``path`` as the data writes it: its generic arguments in
    ``<>`` and the arity its name is keyed by, and a name of ``names`` —
    the parameter names available where it is written — where the
    argument is a parameter."""
    name, args = parts(path)
    held = [a if a in names or "." not in a else written(a, names) for a in args]
    return f"{name}<{', '.join(held)}>" if args else name


def build(origin):
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "dump.ps1"
        script.write_text(DUMP, encoding="utf-8")
        dump = subprocess.run(
            ["pwsh", "-NoProfile", "-File", str(script), str(origin)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    raw = json.loads(dump)
    if isinstance(raw, dict):
        raw = [raw]

    types = {}
    delegates = {}
    deferred_bases = []

    # One name as the data keys it: `Outer`1.Inner`1`.
    def keyed(full):
        return full.replace("+", ".")

    for t in raw:
        name = keyed(t["name"])
        if not _OWNED.match(name):
            continue
        params = t["params"]
        kind = types.setdefault(name, new_type(params))
        if t["base"] and t["base"] != "System.Object":
            deferred_bases.append((kind, t["base"], params))
        deferred_bases.extend(
            (kind, i, params) for i in t["interfaces"] if i is not None
        )
        if t["invoke"]:
            delegates[name] = {
                "params": params,
                "takes": t["invoke"]["takes"],
                "ret": t["invoke"]["ret"],
            }
        for p in t["props"]:
            ret = written(p["ret"], params) if p["ret"] else None
            record(kind["statics" if p["is_static"] else "methods"], p["name"], ret)

    # A base is recorded only where the data holds the type it names.
    for kind, base, params in deferred_bases:
        held = written(base, params)
        if parts(base)[0] in types or held.split("<")[0] in types:
            kind["bases"].append(held)
        # else: a base outside the data is left out

    # An extension method of `this T` is a method of T; the parameters it
    # declares in `this` take the names T declares for them, and a
    # callable it takes is recorded under T as its methods' are.
    callbacks = {}

    def substitute(text, mapping, names):
        """``text`` as source writes it, with the parameter names of
        ``mapping`` renamed, and each of ``names`` left itself."""
        if not text:
            return text
        if not mapping:
            return written(text, names)
        for old, new in mapping.items():
            text = re.sub(rf"\b{re.escape(old)}\b", new, text)
        return written(text, names)

    def callback(arg, names):
        """``{"params", "returns"}`` of the callable passed where the
        parameter ``arg`` of a delegate type is written, or None."""
        delegate, actual = parts(arg)
        held = delegates.get(delegate)
        if not held:
            return None
        mapping = dict(zip(held["params"], actual))
        ret = substitute(held["ret"], mapping, names)
        return {
            "params": [substitute(a, mapping, names) for a in held["takes"]],
            # What stands for the callable's result: a parameter name, or
            # nothing where a concrete type does.
            "returns": ret if ret in names else None,
        }

    for t in raw:
        name = keyed(t["name"])
        if name not in types:
            continue
        for m in t["methods"]:
            args = m["args"] or []
            names = [*t["params"], *m["gparams"]]
            if m["is_extension"]:
                # A method of the type its `this` is of, with that type's
                # parameter names for the ones `this` is written with.
                if not args or not args[0]:
                    continue
                target, given = parts(args[0])
                if target not in types:
                    continue
                if not given or not set(given) <= set(m["gparams"]):
                    # `this` written with a concrete type applies to that
                    # instantiation only, not to every value of the type.
                    continue
                renamed = dict(zip(given, types[target]["params"]))
                for old, new in renamed.items():
                    names = [new if n == old else n for n in names]
                ret = substitute(m["ret"], renamed, names) if m["ret"] else None
                record(types[target]["methods"], m["name"], ret)
                for position, arg in enumerate(args[1:]):
                    held = callback(arg, names)
                    if held:
                        callbacks.setdefault(target, {}).setdefault(m["name"], {})[
                            str(position)
                        ] = {
                            "params": [
                                substitute(a, renamed, names) for a in held["params"]
                            ],
                            "returns": substitute(held["returns"], renamed, names),
                        }
                continue
            ret = written(m["ret"], names) if m["ret"] else None
            record(
                types[name]["statics" if m["is_static"] else "methods"],
                m["name"],
                ret,
            )
            for position, arg in enumerate(args):
                held = arg and callback(arg, names)
                if held:
                    callbacks.setdefault(name, {}).setdefault(m["name"], {})[
                        str(position)
                    ] = held

    # The names a C# file has without any `using`: the language aliases.
    scope = {
        alias: f"System.{name}"
        for alias, name in (
            ("bool", "Boolean"),
            ("byte", "Byte"),
            ("sbyte", "SByte"),
            ("char", "Char"),
            ("decimal", "Decimal"),
            ("double", "Double"),
            ("float", "Single"),
            ("int", "Int32"),
            ("uint", "UInt32"),
            ("nint", "IntPtr"),
            ("nuint", "UIntPtr"),
            ("long", "Int64"),
            ("ulong", "UInt64"),
            ("short", "Int16"),
            ("ushort", "UInt16"),
            ("object", "Object"),
            ("string", "String"),
            ("dynamic", "Object"),
            ("void", "Void"),
        )
    }
    version = _runtime_version(origin)
    return (
        f"the .NET {version} shared runtime, by reflection",
        types,
        {},
        scope,
        None,
        callbacks,
    )


def _runtime_version(origin):
    release = Path(origin) / "Microsoft.NETCore.App.deps.json"
    if release.is_file():
        found = re.search(
            r'"Microsoft\.NETCore\.App/([0-9.]+)', release.read_text("utf-8")
        )
        if found:
            return found[1]
    try:
        out = subprocess.run(
            [
                "pwsh",
                "-NoProfile",
                "-Command",
                "[System.Diagnostics.FileVersionInfo]::GetVersionInfo("
                + f"'{origin}/System.Private.CoreLib.dll').ProductVersion",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out.split("+")[0] or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


if __name__ == "__main__":
    from common import main

    main(build)
