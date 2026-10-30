"""Generate dotnet.json from the .NET shared runtime.

usage: gen_dotnet.py RUNTIME OUTPUT

RUNTIME is the directory that holds the shared runtime's assemblies
(`System.Private.CoreLib.dll` and the `System.*.dll` beside it). The
public and protected types and members of the `System` and `netstandard`
assemblies are read by reflection in PowerShell (`pwsh`), which is the
only thing of the runtime that runs: the `pwsh` in RUNTIME where it has
one, since a process reflects the runtime it runs on. Where a
`*.deps.json` there names the runtime pack an application carries, only
the assemblies of that pack are read, not the packages beside them.

An extension method is written as a method of the type it is written for:
``Enumerable.Where`` is a method of ``IEnumerable`1`` — which is what a
call through an instance of a type that has it as a base resolves to.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import main, new_type, record

DUMP = r"""
$dir = $args[0]
$pack = @(Get-Content $args[1])
$out = @()
foreach ($path in Get-ChildItem $dir -Filter "*.dll") {
    $base = $path.BaseName
    if ($base -ne "netstandard" -and -not $base.StartsWith("System.")) { continue }
    if ($pack.Count -and $pack -notcontains $path.Name) { continue }
    $asm = $null
    try { $asm = [System.Reflection.Assembly]::LoadFrom($path.FullName) } catch {
        try { $asm = [System.Reflection.Assembly]::Load($base) } catch { continue }
    }
    $types = @()
    try { $types = $asm.GetTypes() } catch [System.Reflection.ReflectionTypeLoadException] {
        $types = $_.Exception.Types | Where-Object { $null -ne $_ }
    } catch { continue }
    foreach ($t in $types) {
        # A nested type is seen where every type around it is.
        $seen = $t
        while ($null -ne $seen -and ($seen.IsPublic -or $seen.IsNestedPublic -or $seen.IsNestedFamily -or $seen.IsNestedFamORAssem)) {
            $seen = $seen.DeclaringType
        }
        if ($null -ne $seen) { continue }
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
            byref = $t.IsByRefLike
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


def keyed(full):
    """One name as the data keys it: `Outer`1.Inner`1`."""
    return full.replace("+", ".")


def written(path, names):
    """The type ``path`` as the data writes it: its generic arguments in
    ``<>`` and the arity its name is keyed by, and a name of ``names`` —
    the parameter names available where it is written — where the
    argument is a parameter."""
    name, args = parts(keyed(path))
    held = [a if a in names or "." not in a else written(a, names) for a in args]
    return f"{name}<{', '.join(held)}>" if args else name


def _runtime_pack(origin):
    """The assemblies of the runtime pack an application in ``origin``
    carries, by the ``*.deps.json`` that lists them; none where it has no
    such list."""
    for listing in Path(origin).glob("*.deps.json"):
        for target in json.loads(listing.read_text("utf-8"))["targets"].values():
            for library, held in target.items():
                if library.startswith("runtimepack.Microsoft.NETCore.App."):
                    return [Path(name).name for name in held.get("runtime", ())]
    return []


def sdk_folder(origin, version):
    """The `Sdks` folder of the .NET SDK that bundles the runtime of
    ``version``, beside ``origin`` or any `dotnet` on the path; None for
    none."""
    roots = [Path(origin).resolve().parents[2] / "sdk"] + [
        (Path(folder) / name).resolve().parent / "sdk"
        for folder in os.environ.get("PATH", "").split(os.pathsep)
        for name in ("dotnet", "dotnet.exe")
        if (Path(folder) / name).is_file()
    ]
    bundles = re.compile(r"<BundledNETCoreAppPackageVersion>([^<]+)<")
    return next(
        (
            folder / "Sdks"
            for root in roots
            for folder in sorted(root.glob("*"))
            if (
                props := folder / "Microsoft.NETCoreSdk.BundledVersions.props"
            ).is_file()
            and (found := bundles.search(props.read_text(encoding="utf-8")))
            and found[1] == version
        ),
        None,
    )


def implicit_usings(origin, version):
    """``{SDK name: [[Include or Remove, namespace, conditions]]}``: the
    `<Using>` items of the SDK that bundles the runtime of ``version``,
    beside ``origin`` or the `dotnet` on the path, that its own files
    declare under `<ImplicitUsings>`, in the order MSBuild reads them; each
    holds where its ``conditions``, ``[property, == or !=, value]``, hold
    of the project. Microsoft.NET.Sdk's, with those of the SDKs it imports,
    are every project's; a `Static` or `Alias` item names no namespace."""
    sdk = sdk_folder(origin, version)
    if sdk is None:
        raise SystemExit(f"no .NET SDK bundling the {version} runtime is installed")
    clause = re.compile(r"'\$\((\w+)\)'\s*(==|!=)\s*'([^']*)'")
    found = {}
    for path in sorted(sdk.rglob("*")):
        if path.suffix not in (".props", ".targets") or not path.is_file():
            continue
        text = re.sub(
            r"<!--.*?-->", "", path.read_text("utf-8", "replace"), flags=re.DOTALL
        )
        for group in re.finditer(
            r"<ItemGroup\s+Condition=\"([^\"]*ImplicitUsings[^\"]*)\"\s*>(.*?)</ItemGroup>",
            text,
            re.DOTALL,
        ):
            for item in re.finditer(
                r"<Using\s+(Include|Remove)=\"([^\"]+)\"([^>]*)/>", group[2]
            ):
                if "Static=" in item[3] or "Alias=" in item[3]:
                    continue
                conditions = [
                    list(held)
                    for held in clause.findall(f"{group[1]} {item[3]}")
                    if held[0] != "ImplicitUsings"
                ]
                name = path.relative_to(sdk).parts[0]
                found.setdefault(name, []).append([item[1], item[2], conditions])
    # An SDK that Microsoft.NET.Sdk imports itself (WindowsDesktop) has its
    # items read in every project, after the base's own.
    base = "Microsoft.NET.Sdk"
    imported = {
        name
        for path in (sdk / base).rglob("*")
        if path.suffix in (".props", ".targets") and path.is_file()
        for name in re.findall(
            r"<Import\s+Project=\"\$\(MSBuildThisFileDirectory\)\.\./\.\./([\w.]+)/",
            path.read_text("utf-8", "replace"),
        )
    }
    for name in sorted(imported - {base}):
        found.setdefault(base, []).extend(found.pop(name, ()))
    return found


def build(origin):
    shell = Path(origin) / "pwsh"
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "dump.ps1"
        script.write_text(DUMP, encoding="utf-8")
        pack = Path(tmp) / "pack.txt"
        pack.write_text("\n".join(_runtime_pack(origin)), encoding="utf-8")
        dump = subprocess.run(
            [
                str(shell) if shell.is_file() else "pwsh",
                "-NoProfile",
                "-File",
                str(script),
                str(origin),
                str(pack),
            ],
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
    # Type -> the names of its own instance members; type -> {name of an
    # extension method written for it -> the namespaces that declare one}
    own, extended = {}, {}

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
            if not p["is_static"]:
                own.setdefault(name, set()).add(p["name"])

    # A base is recorded only where the data holds the type it names.
    for kind, base, params in deferred_bases:
        held = written(base, params)
        if parts(base)[0] in types or held.split("<")[0] in types:
            kind["bases"].append(held)

    # An extension method of `this T` is a method of T; the parameters it
    # declares in `this` take the names T declares for them, and a
    # callable it takes is recorded under T as its methods' are.
    callbacks = {}

    def substitute(text, mapping, names):
        """``text`` as source writes it, with the parameter names of
        ``mapping`` renamed, and each of ``names`` left itself."""
        if not text:
            return text
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

    implicit = {}  # type -> the types an operator converts to it implicitly
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
                extended.setdefault(target, {}).setdefault(m["name"], set()).add(
                    name.rpartition(".")[0]
                )
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
            if m["name"] == "op_Implicit" and ret and args and args[0]:
                # A user-defined implicit conversion, to its return type
                implicit.setdefault(ret, set()).add(written(args[0], names))
            if not m["is_static"]:
                own.setdefault(name, set()).add(m["name"])
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
    # A method a type has of its own is called before an extension.
    extensions = {
        target: {
            method: sorted(spaces)
            for method, spaces in names.items()
            if method not in own.get(target, ())
        }
        for target, names in extended.items()
    }
    return {
        "source": f"the .NET {version} shared runtime, by reflection",
        "types": types,
        "functions": {},
        "scope": scope,
        "callbacks": callbacks,
        "extensions": {target: names for target, names in extensions.items() if names},
        "functional": {name: len(held["takes"]) for name, held in delegates.items()},
        "ref_structs": sorted(
            keyed(t["name"]) for t in raw if t["byref"] and _OWNED.match(t["name"])
        ),
        "implicit": {target: sorted(sources) for target, sources in implicit.items()},
        "implicit_usings": implicit_usings(origin, version.split("-")[0]) or None,
    }


def _runtime_version(origin):
    release = Path(origin) / "Microsoft.NETCore.App.deps.json"
    if release.is_file():
        found = re.search(
            r'"Microsoft\.NETCore\.App/([0-9.]+)', release.read_text("utf-8")
        )
        if found:
            return found[1]
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


if __name__ == "__main__":
    main(build)
