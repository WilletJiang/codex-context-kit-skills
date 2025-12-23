#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import os
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path


DEFAULT_PRUNE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "target",
    "dist",
    "build",
    "out",
    "bazel-bin",
    "bazel-out",
    "bazel-testlogs",
    "bazel-workspace",
    ".gradle",
}


@dataclasses.dataclass(frozen=True)
class Sym:
    path: str
    module: str | None
    line: int
    col: int
    kind: str  # class|function|async_function|method|async_method
    qualname: str
    header: str | None
    bases: tuple[str, ...] | None = None
    decorators: tuple[str, ...] | None = None


def _run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return 127, "", f"missing command: {cmd[0]}"
    return proc.returncode, proc.stdout, proc.stderr


def _detect_git_root(path: Path) -> Path | None:
    rc, out, _ = _run(["git", "rev-parse", "--show-toplevel"], cwd=path)
    if rc != 0:
        return None
    p = Path(out.strip())
    return p if p.exists() else None


def _iter_files_non_git(root: Path, max_files: int) -> Iterator[Path]:
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in DEFAULT_PRUNE_DIRS]
        for fname in filenames:
            if not (fname.endswith(".py") or fname.endswith(".pyi")):
                continue
            yield Path(dirpath) / fname
            count += 1
            if count >= max_files:
                return


def _iter_files_git(git_root: Path, rel_from: Path, max_files: int) -> list[Path]:
    rc, out, err = _run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=git_root)
    if rc != 0:
        raise RuntimeError(f"git ls-files failed: {err.strip()}")
    raw = out.split("\0")
    files: list[Path] = []
    for s in raw:
        if not s:
            continue
        if not (s.endswith(".py") or s.endswith(".pyi")):
            continue
        if any(part in DEFAULT_PRUNE_DIRS for part in Path(s).parts[:-1]):
            continue
        p = (git_root / s).resolve()
        try:
            p.relative_to(rel_from)
        except ValueError:
            continue
        files.append(p)
        if len(files) >= max_files:
            break
    return files


def _read_text(path: Path, max_bytes: int) -> str | None:
    try:
        st = path.stat()
    except OSError:
        return None
    if st.st_size > max_bytes:
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _fmt_path(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(p)


def _module_name(path: Path, root: Path) -> str | None:
    # Deterministic, fast approximation: treat `root` as the module root.
    try:
        rel = path.resolve().relative_to(root.resolve())
    except Exception:
        return None
    if rel.suffix not in {".py", ".pyi"}:
        return None
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def _header_line(lines: list[str], lineno: int) -> str | None:
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].rstrip()
    return None


def _decorators(lines: list[str], decorator_list: list[ast.expr]) -> list[str]:
    out: list[str] = []
    for d in decorator_list:
        ln = getattr(d, "lineno", None)
        if ln is None:
            continue
        h = _header_line(lines, int(ln))
        if h:
            out.append(h)
    return out


def _expr_name(expr: ast.AST) -> str | None:
    cur = expr
    if isinstance(cur, ast.Call):
        cur = cur.func
    if isinstance(cur, ast.Subscript):
        cur = cur.value
    if isinstance(cur, ast.Name):
        return cur.id
    if isinstance(cur, ast.Attribute):
        parts: list[str] = []
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            parts.reverse()
            return ".".join(parts)
    return None


def _extract_syms_from_python(
    path: Path,
    *,
    root: Path,
    module_root: Path,
    include_methods: bool,
    include_nested: bool,
    max_bytes: int,
) -> list[Sym]:
    txt = _read_text(path, max_bytes=max_bytes)
    if txt is None:
        return []
    try:
        tree = ast.parse(txt, filename=str(path))
    except SyntaxError:
        return []

    lines = txt.splitlines()
    out: list[Sym] = []
    p = _fmt_path(path)
    mod = _module_name(path, module_root)

    def add(kind: str, qualname: str, node: ast.AST):
        ln = getattr(node, "lineno", None)
        col = getattr(node, "col_offset", 0)
        if ln is None:
            return
        header = _header_line(lines, int(ln))
        bases: tuple[str, ...] | None = None
        decorators: tuple[str, ...] | None = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and getattr(node, "decorator_list", None):
            decs = _decorators(lines, list(node.decorator_list))
            if decs:
                header = "\n".join(decs + ([header] if header else []))
            dec_names = [_expr_name(d) for d in node.decorator_list]
            dec_names = [d for d in dec_names if d]
            if dec_names:
                decorators = tuple(dec_names)
        if isinstance(node, ast.ClassDef):
            base_names = [_expr_name(b) for b in node.bases]
            base_names = [b for b in base_names if b]
            if base_names:
                bases = tuple(base_names)
            dec_names = [_expr_name(d) for d in node.decorator_list]
            dec_names = [d for d in dec_names if d]
            if dec_names:
                decorators = tuple(dec_names)
        out.append(
            Sym(
                path=p,
                module=mod,
                line=int(ln),
                col=int(col) + 1,
                kind=kind,
                qualname=qualname,
                header=header,
                bases=bases,
                decorators=decorators,
            )
        )

    class V(ast.NodeVisitor):
        def __init__(self):
            self.stack: list[tuple[str, str]] = []  # ("class"|"function", name)

        def _qual(self, name: str) -> str:
            if not self.stack:
                return name
            return ".".join([n for _, n in self.stack] + [name])

        def _has_function_scope(self) -> bool:
            return any(k == "function" for k, _ in self.stack)

        def _has_class_scope(self) -> bool:
            return any(k == "class" for k, _ in self.stack)

        def visit_ClassDef(self, node: ast.ClassDef):
            # Default behavior: only top-level classes. Include nested classes when asked.
            if not self.stack or include_nested:
                add("class", self._qual(node.name), node)
            self.stack.append(("class", node.name))
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef):
            in_class = self._has_class_scope()
            in_func = self._has_function_scope()
            if in_func and not include_nested:
                return
            if in_class:
                if include_methods:
                    add("method", self._qual(node.name), node)
            else:
                add("function", self._qual(node.name), node)
            self.stack.append(("function", node.name))
            self.generic_visit(node)
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            in_class = self._has_class_scope()
            in_func = self._has_function_scope()
            if in_func and not include_nested:
                return
            if in_class:
                if include_methods:
                    add("async_method", self._qual(node.name), node)
            else:
                add("async_function", self._qual(node.name), node)
            self.stack.append(("function", node.name))
            self.generic_visit(node)
            self.stack.pop()

    V().visit(tree)

    return out


def build_symbols(
    root: Path,
    *,
    module_root: Path,
    include_methods: bool,
    include_nested: bool,
    query: str | None,
    kinds: set[str] | None,
    bases: list[str] | None,
    decorators: list[str] | None,
    max_files: int,
    max_results: int,
    max_file_bytes: int,
) -> list[Sym]:
    root = root.resolve()
    git_root = _detect_git_root(root)

    files: Iterable[Path]
    if git_root is not None:
        try:
            files = _iter_files_git(git_root, rel_from=root, max_files=max_files)
        except Exception:
            files = _iter_files_non_git(root, max_files=max_files)
    else:
        files = _iter_files_non_git(root, max_files=max_files)

    q = query.casefold() if query else None
    base_q = [b.casefold() for b in (bases or [])]
    deco_q = [d.casefold() for d in (decorators or [])]
    out: list[Sym] = []
    for p in files:
        batch = _extract_syms_from_python(
            p,
            root=root,
            module_root=module_root,
            include_methods=include_methods,
            include_nested=include_nested,
            max_bytes=max_file_bytes,
        )
        if kinds:
            batch = [s for s in batch if s.kind in kinds]
        if q:
            batch = [s for s in batch if q in s.qualname.casefold()]
        if base_q:
            keep: list[Sym] = []
            for s in batch:
                if s.kind != "class":
                    continue
                b = [x.casefold() for x in (s.bases or ())]
                if any(qb in bx for qb in base_q for bx in b):
                    keep.append(s)
            batch = keep
        if deco_q:
            keep2: list[Sym] = []
            for s in batch:
                d = [x.casefold() for x in (s.decorators or ())]
                if any(qd in dx for qd in deco_q for dx in d):
                    keep2.append(s)
            batch = keep2
        if batch:
            out.extend(batch)
        if len(out) >= max_results:
            out = out[:max_results]
            break

    out.sort(key=lambda s: (s.path, s.line, s.col, s.kind, s.qualname))
    return out[:max_results]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="List Python symbols quickly via AST (classes/functions; optional methods).")
    ap.add_argument("path", nargs="?", default=".", help="Search root (default: .).")
    ap.add_argument("--include-methods", action="store_true", help="Include class methods (more output).")
    ap.add_argument("--include-nested", action="store_true", help="Include nested class/function definitions.")
    ap.add_argument("--query", help="Filter by substring match on qualname (case-insensitive).")
    ap.add_argument(
        "--kind",
        action="append",
        default=[],
        help="Filter by kind (repeatable): class,function,async_function,method,async_method.",
    )
    ap.add_argument(
        "--base",
        action="append",
        default=[],
        help="Filter classes by base name substring (repeatable), e.g. nn.Module.",
    )
    ap.add_argument(
        "--decorator",
        action="append",
        default=[],
        help="Filter by decorator name substring (repeatable), e.g. dataclass, torch.no_grad.",
    )
    ap.add_argument(
        "--module-root",
        default=None,
        help="Compute module=... relative to this directory (default: CWD).",
    )
    ap.add_argument("--json", action="store_true", help="Output JSON.")
    ap.add_argument("--banner", action="store_true", help="Print a header line before results.")
    ap.add_argument("--max-files", type=int, default=50_000, help="Cap scanned Python files (default: 50k).")
    ap.add_argument("--max-results", type=int, default=500, help="Cap emitted symbols (default: 500).")
    ap.add_argument(
        "--max-file-bytes",
        type=int,
        default=2_000_000,
        help="Skip files larger than this many bytes (default: 2,000,000).",
    )
    args = ap.parse_args(argv)

    root = Path(args.path)
    if not root.exists():
        print(f"error: path does not exist: {root}", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"error: path is not a directory: {root}", file=sys.stderr)
        return 2
    if args.max_files <= 0 or args.max_results <= 0 or args.max_file_bytes <= 0:
        print("error: max limits must be > 0", file=sys.stderr)
        return 2
    valid_kinds = {"class", "function", "async_function", "method", "async_method"}
    if args.kind:
        bad = [k for k in args.kind if k not in valid_kinds]
        if bad:
            print(f"error: invalid --kind values: {bad}. Valid: {sorted(valid_kinds)}", file=sys.stderr)
            return 2

    include_methods = bool(args.include_methods)
    if args.kind and ({"method", "async_method"} & set(args.kind)):
        include_methods = True
    if args.decorator:
        include_methods = True

    module_root = Path(args.module_root).resolve() if args.module_root else Path.cwd().resolve()

    syms = build_symbols(
        root,
        module_root=module_root,
        include_methods=include_methods,
        include_nested=bool(args.include_nested),
        query=str(args.query) if args.query else None,
        kinds=set(args.kind) if args.kind else None,
        bases=list(args.base) if args.base else None,
        decorators=list(args.decorator) if args.decorator else None,
        max_files=int(args.max_files),
        max_results=int(args.max_results),
        max_file_bytes=int(args.max_file_bytes),
    )

    if args.json:
        payload = [dataclasses.asdict(s) for s in syms]
        print(json.dumps({"root": str(root.resolve()), "count": len(syms), "symbols": payload}, ensure_ascii=False, indent=2))
        return 0

    if not syms:
        if args.query or args.kind or args.base or args.decorator:
            q = f" query={args.query!r}" if args.query else ""
            k = f" kinds={sorted(set(args.kind))}" if args.kind else ""
            b = f" base={args.base!r}" if args.base else ""
            d = f" decorator={args.decorator!r}" if args.decorator else ""
            print(f"sym: no matches ({q}{k}{b}{d})")
        else:
            print("sym: no python symbols found")
        return 1

    if args.banner:
        print(f"# sym (python) count={len(syms)} root={root.resolve()}")
    for s in syms:
        mod = f" module={s.module}" if s.module else ""
        bases = f" bases={list(s.bases)}" if (s.kind == "class" and s.bases) else ""
        decos = f" decorators={list(s.decorators)}" if s.decorators else ""
        print(f"{s.path}:{s.line}:{s.col} kind={s.kind}{mod}{bases}{decos} {s.qualname}")
        if s.header:
            for ln in s.header.splitlines():
                print(f"  {ln}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
