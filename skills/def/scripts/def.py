#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DEPRIORITIZE_SUBSTRS = [
    "/.git/",
    "/.hg/",
    "/.svn/",
    "/.idea/",
    "/.vscode/",
    "/.cache/",
    "/cache/",
    "/tmp/",
    "/log/",
    "/logs/",
    "/sessions/",
    "/artifacts/",
    "/outputs/",
    "/runs/",
    "/wandb/",
    "/mlruns/",
    "/third_party/",
    "/vendor/",
    "/node_modules/",
    "/build/",
    "/dist/",
    "/target/",
    "/.venv/",
    "/venv/",
    "/docs/",
    "/doc/",
    "/examples/",
    "/example/",
    "/test/",
    "/tests/",
    "/bench/",
    "/benchmarks/",
]


LANG_EXTS: dict[str, list[str]] = {
    "py": [".py", ".pyi"],
    "rs": [".rs"],
    "go": [".go"],
    "zig": [".zig"],
    "ts": [".ts", ".tsx"],
    "js": [".js", ".jsx"],
    "c": [".c", ".h"],
    "cpp": [".cc", ".cpp", ".cxx", ".hpp", ".hh"],
    "cuda": [".cu", ".cuh"],
}


@dataclass(frozen=True)
class Candidate:
    path: Path
    line: int
    col: int
    kind: str
    lang: str
    qualname: str | None
    header: str | None
    confidence: str  # "exact" | "high" | "candidate"


def _run_rg_vimgrep(
    pattern: str,
    root: Path,
    *,
    fixed: bool,
    case: str,
    max_hits: int,
    hidden: bool,
    no_ignore: bool,
    globs: list[str],
) -> list[tuple[Path, int, int, str]]:
    cmd: list[str] = ["rg", "--vimgrep", "--color", "never", "--no-messages"]
    if fixed:
        cmd.append("-F")
    if hidden:
        cmd.append("--hidden")
    if no_ignore:
        cmd.append("--no-ignore")
    if case == "sensitive":
        cmd.append("--case-sensitive")
    elif case == "insensitive":
        cmd.append("--ignore-case")
    else:
        cmd.append("--smart-case")
    for g in globs:
        cmd.extend(["--glob", g])
    cmd.extend([pattern, str(root)])

    hits: list[tuple[Path, int, int, str]] = []
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        raise RuntimeError("ripgrep (rg) not found on PATH")

    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            parts = line.rstrip("\n").split(":", 3)
            if len(parts) != 4:
                continue
            p, ln, col, txt = parts
            try:
                hits.append((Path(p), int(ln), int(col), txt))
            except ValueError:
                continue
            if len(hits) >= max_hits:
                proc.terminate()
                break
    finally:
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
    return hits


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


def _read_lines(path: Path, max_bytes: int) -> list[str] | None:
    txt = _read_text(path, max_bytes=max_bytes)
    if txt is None:
        return None
    return txt.splitlines()


def _fmt_path(p: Path) -> str:
    try:
        rel = p.resolve().relative_to(Path(os.getcwd()).resolve())
        return str(rel)
    except Exception:
        return str(p)


def _path_score(path: Path, *, prefer: list[str]) -> int:
    s = _fmt_path(path).replace(os.sep, "/")
    wrapped = f"/{s}/"
    penalty = 0
    for sub in DEFAULT_DEPRIORITIZE_SUBSTRS:
        if sub in wrapped:
            penalty += 50
    for pat in prefer:
        if pat and pat in s:
            penalty -= 100
    return penalty


def _extract_snippet(lines: list[str], line: int, context: int) -> list[str]:
    if context <= 0:
        return [f"L{line}: {lines[line - 1]}"] if 1 <= line <= len(lines) else []
    s = max(1, line - context)
    e = min(len(lines), line + context)
    out: list[str] = []
    for ln in range(s, e + 1):
        out.append(f"L{ln}: {lines[ln - 1]}")
    return out


class _PyQualVisitor(ast.NodeVisitor):
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.stack: list[str] = []
        self.matches: list[tuple[ast.AST, str]] = []

    def _push(self, name: str):
        self.stack.append(name)

    def _pop(self):
        self.stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef):
        self._push(node.name)
        if node.name == self.symbol:
            self.matches.append((node, ".".join(self.stack)))
        self.generic_visit(node)
        self._pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._push(node.name)
        if node.name == self.symbol:
            self.matches.append((node, ".".join(self.stack)))
        self.generic_visit(node)
        self._pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._push(node.name)
        if node.name == self.symbol:
            self.matches.append((node, ".".join(self.stack)))
        self.generic_visit(node)
        self._pop()


def _python_defs_in_file(path: Path, symbol: str, *, max_bytes: int) -> list[Candidate]:
    txt = _read_text(path, max_bytes=max_bytes)
    if txt is None:
        return []
    try:
        tree = ast.parse(txt, filename=str(path))
    except SyntaxError:
        return []

    lines = txt.splitlines()
    visitor = _PyQualVisitor(symbol)
    visitor.visit(tree)

    out: list[Candidate] = []
    for node, qual in visitor.matches:
        lineno = getattr(node, "lineno", None)
        col = getattr(node, "col_offset", 0)
        if lineno is None:
            continue

        header = lines[lineno - 1].rstrip() if 1 <= lineno <= len(lines) else None
        # include decorators if present (only for functions)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and getattr(node, "decorator_list", None):
            dec_lines: list[str] = []
            for dec in node.decorator_list:
                dl = getattr(dec, "lineno", None)
                if dl is None or not (1 <= dl <= len(lines)):
                    continue
                dec_lines.append(lines[dl - 1].rstrip())
            if dec_lines:
                header = "\n".join(dec_lines + ([header] if header else []))

        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        out.append(
            Candidate(
                path=Path(_fmt_path(path)),
                line=int(lineno),
                col=int(col) + 1,  # make it 1-based for humans
                kind=kind,
                lang="py",
                qualname=qual,
                header=header,
                confidence="exact",
            )
        )
    return out


def _rg_globs_for_lang(lang: str) -> list[str]:
    exts = LANG_EXTS.get(lang, [])
    return [f"*{e}" for e in exts]


def _python_find_candidate_files(
    symbol: str,
    root: Path,
    *,
    max_hits: int,
    hidden: bool,
    no_ignore: bool,
    case: str,
) -> list[Path]:
    # Anchored patterns to keep results small. Most Python defs have def/class on first line.
    pat = rf"^\s*(async\s+def|def|class)\s+{re.escape(symbol)}\b"
    hits = _run_rg_vimgrep(
        pat,
        root,
        fixed=False,
        case=case,
        max_hits=max_hits,
        hidden=hidden,
        no_ignore=no_ignore,
        globs=_rg_globs_for_lang("py"),
    )
    files: list[Path] = []
    seen: set[Path] = set()
    for p, _, _, _ in hits:
        if p in seen:
            continue
        seen.add(p)
        files.append(p)
    return files


def _heuristic_candidates(
    symbol: str,
    root: Path,
    *,
    lang: str,
    max_hits: int,
    hidden: bool,
    no_ignore: bool,
    case: str,
) -> list[Candidate]:
    sym = re.escape(symbol)
    patterns: list[tuple[str, str]] = []

    if lang == "rs":
        patterns = [
            (rf"^\s*(pub(\([^)]*\))?\s+)?(unsafe\s+)?(async\s+)?fn\s+{sym}\b", "function"),
            (rf"^\s*(pub\s+)?(struct|enum|trait|type)\s+{sym}\b", "type"),
        ]
    elif lang == "go":
        patterns = [
            (rf"^\s*func\s+(\([^)]+\)\s*)?{sym}\s*\(", "function"),
            (rf"^\s*type\s+{sym}\s+(struct|interface)\b", "type"),
        ]
    elif lang == "zig":
        patterns = [
            (rf"^\s*(pub\s+)?fn\s+{sym}\b", "function"),
            (rf"^\s*(pub\s+)?(const|var)\s+{sym}\b", "value"),
        ]
    elif lang in {"ts", "js"}:
        patterns = [
            (rf"^\s*(export\s+)?(default\s+)?(async\s+)?function\s+{sym}\b", "function"),
            (rf"^\s*(export\s+)?class\s+{sym}\b", "class"),
            (rf"^\s*(export\s+)?(const|let|var)\s+{sym}\s*=\s*\(", "function-ish"),
            (rf"^\s*(export\s+)?(const|let|var)\s+{sym}\s*=\s*async\s*\(", "function-ish"),
        ]
    elif lang in {"c", "cpp", "cuda"}:
        patterns = [
            (rf"^\s*(class|struct|enum)\s+{sym}\b", "type"),
            (rf"^\s*#\s*define\s+{sym}\b", "macro"),
            # function definition-ish: name( ... ) and line ends with '{' (common case)
            (rf"^\s*[^\n]*\b{sym}\s*\([^;]*\)\s*\{{\s*$", "function-ish"),
        ]
    else:
        return []

    globs = _rg_globs_for_lang(lang)
    out: list[Candidate] = []
    for pat, kind in patterns:
        hits = _run_rg_vimgrep(
            pat,
            root,
            fixed=False,
            case=case,
            max_hits=max_hits,
            hidden=hidden,
            no_ignore=no_ignore,
            globs=globs,
        )
        for p, ln, col, txt in hits:
            out.append(
                Candidate(
                    path=Path(_fmt_path((root / p).resolve() if not p.is_absolute() else p)),
                    line=int(ln),
                    col=int(col),
                    kind=kind,
                    lang=lang,
                    qualname=None,
                    header=txt.rstrip(),
                    confidence="candidate",
                )
            )
        if len(out) >= max_hits:
            break
    return out[:max_hits]


def _dedupe(cands: list[Candidate]) -> list[Candidate]:
    seen: set[tuple[str, int, str, str]] = set()
    out: list[Candidate] = []
    for c in cands:
        key = (str(c.path), c.line, c.kind, c.lang)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Locate symbol definitions (Python precise; others conservative).")
    ap.add_argument("symbol", help="Symbol name to locate (identifier, not regex).")
    ap.add_argument("path", nargs="?", default=".", help="Repo root/search root (default: .).")
    ap.add_argument(
        "--lang",
        default="auto",
        choices=["auto", "py", "rs", "go", "zig", "ts", "js", "c", "cpp", "cuda"],
        help="Language mode (default: auto).",
    )
    ap.add_argument("--json", action="store_true", help="Output JSON.")
    ap.add_argument("-C", "--context", type=int, default=0, help="Context lines around each hit (default: 0).")
    ap.add_argument("--max-results", type=int, default=30, help="Cap total results (default: 30).")
    ap.add_argument("--max-hits", type=int, default=80, help="Cap rg hits per phase (default: 80).")
    ap.add_argument(
        "--max-file-bytes",
        type=int,
        default=2_000_000,
        help="Skip files larger than this many bytes (default: 2,000,000).",
    )
    ap.add_argument("--hidden", action="store_true", help="Search hidden files/dirs (rg --hidden).")
    ap.add_argument("--no-ignore", action="store_true", help="Do not respect ignore files (rg --no-ignore).")
    ap.add_argument(
        "--case",
        choices=["smart", "insensitive", "sensitive"],
        default="smart",
        help="Case handling for non-Python candidate search (default: smart).",
    )
    ap.add_argument(
        "--prefer",
        action="append",
        default=[],
        help="Prefer paths containing this substring (repeatable).",
    )
    args = ap.parse_args(argv)

    symbol = args.symbol.strip()
    if not symbol or not re.match(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$", symbol):
        print("error: symbol must be an identifier or dotted qualname like Foo, foo_bar, or A.foo", file=sys.stderr)
        return 2
    qual_filter = symbol if "." in symbol else None
    base_symbol = symbol.split(".")[-1]
    if symbol == "SYMBOL":
        # Common copy/paste placeholder; make the failure mode obvious.
        print("note: replace the literal placeholder SYMBOL with a real identifier (e.g. MyTrainer)")
    if args.context < 0:
        print("error: --context must be >= 0", file=sys.stderr)
        return 2
    if args.max_results <= 0 or args.max_hits <= 0 or args.max_file_bytes <= 0:
        print("error: max limits must be > 0", file=sys.stderr)
        return 2

    root = Path(args.path).resolve()
    if not root.exists():
        print(f"error: path does not exist: {root}", file=sys.stderr)
        return 2
    search_root = root if root.is_dir() else root.parent

    cands: list[Candidate] = []

    def add_sorted(items: list[Candidate]):
        nonlocal cands
        cands.extend(items)
        cands = _dedupe(cands)
        cands.sort(
            key=lambda c: (
                0 if c.confidence == "exact" else 1,
                _path_score(Path(str(c.path)), prefer=list(args.prefer)),
                0 if (c.qualname and "." not in c.qualname) else 1,
                str(c.path),
                c.line,
            )
        )
        cands = cands[: int(args.max_results)]

    if args.lang in {"auto", "py"}:
        # Phase 1: shortlist files via rg, then AST-validate definitions.
        try:
            files = _python_find_candidate_files(
                base_symbol,
                search_root,
                max_hits=int(args.max_hits),
                hidden=bool(args.hidden),
                no_ignore=bool(args.no_ignore),
                case=str(args.case),
            )
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

        # Resolve paths relative to search root.
        abs_files: list[Path] = []
        for p in files:
            abs_files.append((search_root / p).resolve() if not p.is_absolute() else p.resolve())

        py_defs: list[Candidate] = []
        for p in abs_files:
            py_defs.extend(_python_defs_in_file(p, base_symbol, max_bytes=int(args.max_file_bytes)))
            if len(py_defs) >= int(args.max_results):
                break
        if qual_filter is not None:
            py_defs = [c for c in py_defs if (c.qualname or "").endswith(qual_filter)]
        add_sorted(py_defs)

        if args.lang == "py":
            pass

    # If not enough exact results or explicitly asked for other languages, add candidates.
    if args.lang != "py" and (args.lang != "auto" or len([c for c in cands if c.confidence == "exact"]) == 0):
        langs = (
            [args.lang]
            if args.lang != "auto"
            else ["rs", "go", "zig", "ts", "js", "c", "cpp", "cuda"]
        )
        for lang in langs:
            try:
                add_sorted(
                    _heuristic_candidates(
                        symbol,
                        search_root,
                        lang=lang,
                        max_hits=int(args.max_hits),
                        hidden=bool(args.hidden),
                        no_ignore=bool(args.no_ignore),
                        case=str(args.case),
                    )
                )
            except RuntimeError as e:
                print(f"error: {e}", file=sys.stderr)
                return 2

    if args.json:
        payload = [
            {
                "path": str(c.path),
                "line": c.line,
                "col": c.col,
                "lang": c.lang,
                "kind": c.kind,
                "qualname": c.qualname,
                "confidence": c.confidence,
                "header": c.header,
            }
            for c in cands
        ]
        print(json.dumps({"symbol": symbol, "results": payload}, ensure_ascii=False, indent=2))
        return 0 if cands else 1

    if not cands:
        print(f"def: no definitions found for `{symbol}` under `{search_root}` (lang={args.lang})")
        print("hint: pick a real identifier (e.g. a class/function name) and try again; use `ctx` or `rg '^class '` to discover candidates.")
        return 1

    for c in cands:
        q = f" qualname={c.qualname}" if c.qualname else ""
        emit_hdr = c.header if c.header else ""
        print(f"{c.path}:{c.line}:{c.col} lang={c.lang} kind={c.kind} confidence={c.confidence}{q}")
        if emit_hdr:
            for ln in emit_hdr.splitlines():
                print(f"  {ln}")
        if args.context > 0:
            p = Path(str(c.path))
            abs_p = p if p.is_absolute() else (Path(os.getcwd()).resolve() / p).resolve()
            lines = _read_lines(abs_p, max_bytes=int(args.max_file_bytes))
            if lines is None:
                print(f"  (snippet skipped: unreadable or >{int(args.max_file_bytes)} bytes)")
            else:
                for s in _extract_snippet(lines, c.line, int(args.context)):
                    print(f"  {s}")
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
