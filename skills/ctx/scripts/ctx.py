#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Hit:
    path: Path
    line: int
    col: int
    text: str


EXT_LANG = {
    ".py": "py",
    ".pyi": "py",
    ".rs": "rs",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cu": "cuda",
    ".cuh": "cuda",
    ".go": "go",
    ".ts": "ts",
    ".tsx": "ts",
    ".js": "js",
    ".jsx": "js",
    ".zig": "zig",
}

# Only affects ordering, not inclusion.
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


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)


def _run_rg_stream(
    pattern: str,
    cwd: Path,
    targets: list[str],
    *,
    fixed: bool,
    hidden: bool,
    no_ignore: bool,
    case: str,
    max_hits: int,
    globs: list[str],
) -> list[Hit]:
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
    cmd.append(pattern)
    cmd.extend(targets)

    hits: list[Hit] = []
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
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
            # vimgrep: file:line:col:matchline
            parts = line.rstrip("\n").split(":", 3)
            if len(parts) != 4:
                continue
            p, ln, col, txt = parts
            try:
                hit = Hit(path=Path(p), line=int(ln), col=int(col), text=txt)
            except ValueError:
                continue
            hits.append(hit)
            if len(hits) >= max_hits:
                proc.terminate()
                break
    finally:
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()

    return hits


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals.sort()
    merged: list[tuple[int, int]] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e + 1:
            cur_e = max(cur_e, e)
            continue
        merged.append((cur_s, cur_e))
        cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _read_lines(path: Path, max_bytes: int) -> list[str] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    if st.st_size > max_bytes:
        return None
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # keepends False
    return data.splitlines()


def _detect_lang(path: Path) -> str:
    return EXT_LANG.get(path.suffix.lower(), "other")


def _lang_penalty(path: Path) -> int:
    ext = path.suffix.lower()
    if ext in EXT_LANG:
        return 0
    if ext in {".toml", ".yaml", ".yml", ".ini", ".cfg", ".json"}:
        return 10
    if ext in {".md", ".rst", ".txt"}:
        return 15
    if ext in {".jsonl", ".csv", ".tsv", ".parquet"}:
        return 25
    return 30


def _scope_for_anchor(lines: list[str], anchor: int, *, lang: str, lookback: int) -> list[str]:
    # anchor is 1-based.
    if anchor < 1 or anchor > len(lines):
        return []
    start = max(1, anchor - lookback)

    if lang == "py":
        pat = re.compile(r"^\s*(async\s+def|def|class)\s+[A-Za-z_]\w*\b")
        deco = re.compile(r"^\s*@")
        anchor_line = lines[anchor - 1]
        anchor_indent = len(anchor_line) - len(anchor_line.lstrip(" \t"))
        for ln in range(anchor, start - 1, -1):
            s = lines[ln - 1]
            if pat.match(s):
                hdr_indent = len(s) - len(s.lstrip(" \t"))
                if hdr_indent > anchor_indent:
                    continue
                out = [s.rstrip()]
                k = ln - 1
                while k >= 1 and deco.match(lines[k - 1]):
                    out.insert(0, lines[k - 1].rstrip())
                    k -= 1
                    if len(out) >= 5:
                        break
                return out
        return []

    if lang == "rs":
        pat = re.compile(
            r"^\s*(pub(\([^)]*\))?\s+)?(unsafe\s+)?(async\s+)?(fn|struct|enum|trait|impl|type|mod)\s+\w+"
        )
        for ln in range(anchor, start - 1, -1):
            s = lines[ln - 1]
            if pat.match(s):
                return [s.rstrip()]
        return []

    if lang == "go":
        pat = re.compile(r"^\s*func\s+(\([^)]+\)\s*)?[A-Za-z_]\w*\s*\(")
        for ln in range(anchor, start - 1, -1):
            s = lines[ln - 1]
            if pat.match(s):
                return [s.rstrip()]
        return []

    if lang in {"ts", "js"}:
        pat = re.compile(
            r"^\s*(export\s+)?(default\s+)?(async\s+)?(function\s+[A-Za-z_]\w*\s*\(|class\s+[A-Za-z_]\w*\b|interface\s+[A-Za-z_]\w*\b|type\s+[A-Za-z_]\w*\b)"
        )
        for ln in range(anchor, start - 1, -1):
            s = lines[ln - 1]
            if pat.match(s):
                return [s.rstrip()]
        return []

    if lang == "zig":
        pat = re.compile(r"^\s*(pub\s+)?(fn|const|var)\s+[A-Za-z_]\w*\b")
        for ln in range(anchor, start - 1, -1):
            s = lines[ln - 1]
            if pat.match(s):
                return [s.rstrip()]
        return []

    if lang in {"c", "cpp", "cuda"}:
        for ln in range(anchor, start - 1, -1):
            s = lines[ln - 1].rstrip()
            if not s or s.lstrip().startswith(("//", "/*", "*", "#")):
                continue
            if "(" not in s:
                continue
            if s.endswith("{") or s.endswith(")") or s.endswith("){") or s.endswith(") {"):
                if re.match(r"^\s*(if|for|while|switch|return)\b", s):
                    continue
                if len(s) > 220:
                    continue
                return [s]
        return []

    return []


def _path_score(
    path: Path,
    *,
    hits: int,
    prefer: list[str],
    deprioritize_substrs: list[str],
) -> tuple[int, int, int, str]:
    s = str(path).replace(os.sep, "/")
    wrapped = f"/{s}/"
    penalty = 0
    for sub in deprioritize_substrs:
        if sub in wrapped:
            penalty += 50

    for pat in prefer:
        if pat and pat in s:
            penalty -= 100

    return (penalty + _lang_penalty(path), -hits, len(s), s)


def _print_packet(
    *,
    hits: list[Hit],
    context: int,
    max_files: int,
    max_output_lines: int,
    max_file_bytes: int,
    show_line_numbers: bool,
    show_scope: bool,
    scope_lookback: int,
    prefer: list[str],
    as_json: bool,
) -> int:
    if not hits:
        if as_json:
            print(
                json_dumps(
                    {
                        "truncated": False,
                        "files": [],
                        "error": "no matches",
                    }
                )
            )
        else:
            print("ctx: no matches")
        return 1

    by_file: dict[Path, list[Hit]] = {}
    for h in hits:
        if h.path not in by_file:
            if len(by_file) >= max_files:
                continue
            by_file[h.path] = []
        by_file[h.path].append(h)

    sorted_items = sorted(
        by_file.items(),
        key=lambda kv: _path_score(
            kv[0],
            hits=len(kv[1]),
            prefer=prefer,
            deprioritize_substrs=DEFAULT_DEPRIORITIZE_SUBSTRS,
        ),
    )

    out_lines_left = max_output_lines
    truncated = False

    def emit(s: str) -> None:
        nonlocal out_lines_left
        if out_lines_left <= 0:
            return
        print(s)
        out_lines_left -= 1

    def take(n: int) -> bool:
        nonlocal out_lines_left, truncated
        if out_lines_left <= 0:
            truncated = True
            return False
        if n <= out_lines_left:
            out_lines_left -= n
            return True
        out_lines_left = 0
        truncated = True
        return False

    packet: dict[str, Any] = {"truncated": False, "files": []}

    for path, file_hits in sorted_items:
        lines = _read_lines(path, max_bytes=max_file_bytes)
        if lines is None:
            if as_json:
                packet["files"].append(
                    {
                        "path": str(path),
                        "skipped": True,
                        "reason": f"unreadable or >{max_file_bytes} bytes",
                    }
                )
            else:
                emit(f"===== {path} (skipped: unreadable or >{max_file_bytes} bytes) =====")
                emit("")
            continue

        match_lines = sorted({h.line for h in file_hits if h.line >= 1})
        intervals = []
        for ln in match_lines:
            s = max(1, ln - context)
            e = min(len(lines), ln + context)
            intervals.append((s, e))
        intervals = _merge_intervals(intervals)

        if as_json:
            file_entry: dict[str, Any] = {
                "path": str(path),
                "skipped": False,
                "hits": len(file_hits),
                "chunks": [],
            }
        else:
            emit(f"===== {path} (hits: {len(file_hits)}, chunks: {len(intervals)}) =====")

        lang = _detect_lang(path)
        for (s, e) in intervals:
            if out_lines_left <= 0:
                break
            # chunk header references first match line within chunk, if any
            anchor = next((ln for ln in match_lines if s <= ln <= e), s)
            anchor_cols = [h.col for h in file_hits if h.line == anchor and h.col >= 1]
            anchor_col = min(anchor_cols) if anchor_cols else 1
            if as_json:
                chunk: dict[str, Any] = {
                    "anchor": {"line": anchor, "col": anchor_col},
                    "range": {"start": s, "end": e},
                    "scope": [],
                    "lines": [],
                }
            else:
                emit(f"{path}:{anchor}:{anchor_col}")
            if show_scope:
                scope = _scope_for_anchor(lines, anchor, lang=lang, lookback=scope_lookback)
                for sl in scope:
                    if as_json:
                        if not take(1):
                            break
                        chunk["scope"].append(sl)
                    else:
                        if out_lines_left <= 0:
                            break
                        emit(f"SCOPE: {sl}")
            for ln in range(s, e + 1):
                if out_lines_left <= 0:
                    break
                content = lines[ln - 1]
                if as_json:
                    if not take(1):
                        break
                    chunk["lines"].append({"line": ln, "text": content})
                else:
                    if show_line_numbers:
                        emit(f"L{ln}: {content}")
                    else:
                        emit(content)
            if as_json:
                file_entry["chunks"].append(chunk)
            else:
                emit("")

        if out_lines_left <= 0:
            break

        if as_json:
            packet["files"].append(file_entry)

    if as_json:
        packet["truncated"] = bool(truncated)
        print(json_dumps(packet))
        return 2 if truncated else 0

    if out_lines_left <= 0:
        print("... (ctx truncated by --max-output-lines)")
        return 2
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Extract bounded context around rg matches (fast, file-read once, merged windows)."
    )
    ap.add_argument("pattern", help="rg pattern (regex by default; use -F for literal).")
    ap.add_argument("path", nargs="?", default=".", help="Search root path (default: .).")
    ap.add_argument("-F", "--fixed", action="store_true", help="Treat pattern as a literal string.")
    ap.add_argument("-C", "--context", type=int, default=3, help="Context lines before/after (default: 3).")
    ap.add_argument("--max-hits", type=int, default=30, help="Max rg hits to collect before stopping.")
    ap.add_argument("--max-files", type=int, default=20, help="Max distinct files to include.")
    ap.add_argument(
        "--max-output-lines",
        type=int,
        default=400,
        help="Max output lines to print (bounded packet).",
    )
    ap.add_argument(
        "--max-file-bytes",
        type=int,
        default=2_000_000,
        help="Skip files larger than this many bytes (default: 2,000,000).",
    )
    ap.add_argument(
        "--case",
        choices=["smart", "insensitive", "sensitive"],
        default="smart",
        help="Case handling (default: smart).",
    )
    ap.add_argument("--hidden", action="store_true", help="Search hidden files/dirs (rg --hidden).")
    ap.add_argument("--no-ignore", action="store_true", help="Do not respect ignore files (rg --no-ignore).")
    ap.add_argument("--glob", action="append", default=[], help="Additional rg --glob pattern (repeatable).")
    ap.add_argument("--no-line-numbers", action="store_true", help="Do not prefix lines with L<line>:")
    ap.add_argument("--json", action="store_true", help="Output JSON instead of text.")
    scope_g = ap.add_mutually_exclusive_group()
    scope_g.add_argument("--scope", action="store_true", help="Print a scope header above each chunk anchor.")
    scope_g.add_argument("--no-scope", action="store_true", help="Disable scope headers (fastest).")
    ap.add_argument(
        "--scope-lookback",
        type=int,
        default=200,
        help="Max lines to search upward for scope header (default: 200).",
    )
    ap.add_argument(
        "--prefer",
        action="append",
        default=[],
        help="Prefer paths containing this substring (repeatable).",
    )
    args = ap.parse_args(argv)

    if args.context < 0:
        print("error: --context must be >= 0", file=sys.stderr)
        return 2
    if args.scope_lookback <= 0:
        print("error: --scope-lookback must be > 0", file=sys.stderr)
        return 2
    if args.max_hits <= 0 or args.max_files <= 0 or args.max_output_lines <= 0 or args.max_file_bytes <= 0:
        print("error: max limits must be > 0", file=sys.stderr)
        return 2

    in_path = Path(args.path).resolve()
    if not in_path.exists():
        print(f"error: path does not exist: {in_path}", file=sys.stderr)
        return 2

    # Search target(s): directory or a single file.
    if in_path.is_dir():
        search_cwd = in_path
        targets = [str(in_path)]
    else:
        search_cwd = in_path.parent
        targets = [str(in_path)]

    try:
        hits = _run_rg_stream(
            args.pattern,
            search_cwd,
            targets,
            fixed=bool(args.fixed),
            hidden=bool(args.hidden),
            no_ignore=bool(args.no_ignore),
            case=str(args.case),
            max_hits=int(args.max_hits),
            globs=list(args.glob),
        )
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # Normalize hit paths: rg prints paths relative to search root by default; keep them relative.
    norm_hits: list[Hit] = []
    for h in hits:
        p = h.path
        if not p.is_absolute():
            p = (search_cwd / p).resolve()
        # keep within filesystem bounds (avoid weirdness)
        norm_hits.append(Hit(path=p, line=h.line, col=h.col, text=h.text))

    # Make output stable and locally-scoped when possible (relative paths if under CWD)
    cwd = Path(os.getcwd()).resolve()
    stable_hits: list[Hit] = []
    for h in norm_hits:
        try:
            rel = h.path.relative_to(cwd)
            stable_hits.append(Hit(path=rel, line=h.line, col=h.col, text=h.text))
        except ValueError:
            stable_hits.append(h)

    return _print_packet(
        hits=stable_hits,
        context=int(args.context),
        max_files=int(args.max_files),
        max_output_lines=int(args.max_output_lines),
        max_file_bytes=int(args.max_file_bytes),
        show_line_numbers=not bool(args.no_line_numbers),
        show_scope=(False if args.no_scope else True),
        scope_lookback=int(args.scope_lookback),
        prefer=list(args.prefer),
        as_json=bool(args.json),
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
