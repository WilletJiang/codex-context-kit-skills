#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import fnmatch
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator


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

DEFAULT_PRUNE_GLOBS = {
    ".DS_Store",
    "*.pyc",
    "*.pyo",
    "*.o",
    "*.obj",
    "*.a",
    "*.lib",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.exe",
    "*.bin",
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.tgz",
    "*.7z",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.pdf",
    "*.onnx",
    "*.pt",
    "*.pth",
    "*.ckpt",
}


EXT_LANGUAGE = {
    ".py": "Python",
    ".pyi": "Python",
    ".rs": "Rust",
    ".c": "C",
    ".h": "C/C++ Header",
    ".cc": "C++",
    ".cpp": "C++",
    ".cxx": "C++",
    ".hpp": "C/C++ Header",
    ".hh": "C/C++ Header",
    ".cu": "CUDA",
    ".cuh": "CUDA",
    ".hip": "HIP",
    ".mm": "ObjC++",
    ".m": "ObjC",
    ".java": "Java",
    ".kt": "Kotlin",
    ".go": "Go",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".json": "JSON",
    ".toml": "TOML",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".md": "Markdown",
    ".sh": "Shell",
    ".bash": "Shell",
    ".zsh": "Shell",
    ".ps1": "PowerShell",
    ".cmake": "CMake",
    ".mak": "Make",
    ".mk": "Make",
    ".ini": "INI",
    ".cfg": "Config",
}


TOPLEVEL_SIGNAL_FILES = [
    # Build systems / language toolchains
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "setup.py",
    "setup.cfg",
    "Pipfile",
    "poetry.lock",
    "uv.lock",
    "environment.yml",
    "Cargo.toml",
    "Cargo.lock",
    "CMakeLists.txt",
    "Makefile",
    "meson.build",
    "BUILD.bazel",
    "WORKSPACE",
    "WORKSPACE.bazel",
    "MODULE.bazel",
    "package.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "bun.lockb",
    "go.mod",
    "go.sum",
    "gradlew",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    # Tests / CI
    "pytest.ini",
    "tox.ini",
    ".coveragerc",
    "conftest.py",
    ".github/workflows",
    ".gitlab-ci.yml",
    "azure-pipelines.yml",
    ".circleci/config.yml",
    # Docs / meta
    "README.md",
    "README.rst",
    "CONTRIBUTING.md",
    "LICENSE",
    "NOTICE",
]

INTERESTING_FILENAMES = {
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "setup.py",
    "setup.cfg",
    "Pipfile",
    "poetry.lock",
    "uv.lock",
    "environment.yml",
    "Cargo.toml",
    "Cargo.lock",
    "CMakeLists.txt",
    "Makefile",
    "meson.build",
    "BUILD.bazel",
    "WORKSPACE",
    "WORKSPACE.bazel",
    "MODULE.bazel",
    "package.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "bun.lockb",
    "go.mod",
    "go.sum",
    "gradlew",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "pytest.ini",
    "tox.ini",
    ".coveragerc",
    ".gitignore",
    ".gitattributes",
    "README.md",
    "README.rst",
    "CONTRIBUTING.md",
    "LICENSE",
    "NOTICE",
}

PRIORITY_DIRS = {
    "src",
    "include",
    "python",
    "cpp",
    "csrc",
    "torch",
    "aten",
    "c10",
    "test",
    "tests",
    "benchmarks",
    "examples",
    "scripts",
    "tools",
    "third_party",
    "vendor",
    "docs",
}


@dataclasses.dataclass(frozen=True)
class GitInfo:
    is_repo: bool
    root: str | None = None
    head: str | None = None
    branch: str | None = None
    dirty: bool | None = None


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


def detect_git(path: Path) -> GitInfo:
    rc, out, _ = _run(["git", "rev-parse", "--is-inside-work-tree"], cwd=path)
    if rc != 0 or out.strip() != "true":
        return GitInfo(is_repo=False)

    rc, root, _ = _run(["git", "rev-parse", "--show-toplevel"], cwd=path)
    git_root = root.strip() if rc == 0 else str(path.resolve())
    root_path = Path(git_root)

    head = None
    branch = None
    dirty = None

    rc, head_out, _ = _run(["git", "rev-parse", "HEAD"], cwd=root_path)
    if rc == 0:
        head = head_out.strip()

    rc, br_out, _ = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root_path)
    if rc == 0:
        b = br_out.strip()
        branch = None if b == "HEAD" else b

    rc, st_out, _ = _run(["git", "status", "--porcelain=v1"], cwd=root_path)
    if rc == 0:
        dirty = bool(st_out.strip())

    return GitInfo(is_repo=True, root=str(root_path), head=head, branch=branch, dirty=dirty)


def _should_prune_dir(name: str, prune_dirs: set[str]) -> bool:
    return name in prune_dirs


def _should_prune_file(name: str, prune_globs: set[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in prune_globs)


def iter_files_non_git(
    root: Path,
    prune_dirs: set[str],
    prune_globs: set[str],
    max_files: int,
) -> Iterator[Path]:
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # prune directories in-place
        dirnames[:] = [d for d in dirnames if not _should_prune_dir(d, prune_dirs)]

        for fname in filenames:
            if _should_prune_file(fname, prune_globs):
                continue
            p = Path(dirpath) / fname
            yield p
            count += 1
            if count >= max_files:
                return


def iter_files_git(
    git_root: Path,
    rel_from: Path,
    *,
    prune_dirs: set[str],
    prune_globs: set[str],
    max_files: int,
) -> list[Path]:
    # Enumerate tracked + untracked (but not ignored) files.
    #
    # Why include untracked?
    # - fresh clones / new repos may have nothing "tracked" yet
    # - many workflows stage/commit later, but you still want correct repo facts now
    cmd = ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    rc, out, err = _run(cmd, cwd=git_root)
    if rc != 0:
        raise RuntimeError(f"git ls-files failed: {err.strip()}")
    raw = out.split("\0")
    paths: list[Path] = []
    for s in raw:
        if not s:
            continue
        p = git_root / s
        # If user passed a subdir, restrict to it.
        try:
            p.relative_to(rel_from)
        except ValueError:
            continue
        try:
            rel = p.resolve().relative_to(rel_from.resolve())
        except Exception:
            continue
        if any(part in prune_dirs for part in rel.parts[:-1]):
            continue
        if _should_prune_file(rel.name, prune_globs):
            continue
        paths.append(p)
        if len(paths) >= max_files:
            break
    return paths


def _language_for_path(p: Path) -> str:
    ext = p.suffix.lower()
    if ext in EXT_LANGUAGE:
        return EXT_LANGUAGE[ext]
    # Treat some extensionless build files specially.
    if p.name in {"Makefile", "Dockerfile"}:
        return p.name
    return "Other"


def _safe_stat(p: Path):
    try:
        return p.stat()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _find_signal_paths(root: Path) -> list[str]:
    hits: list[str] = []
    for item in TOPLEVEL_SIGNAL_FILES:
        target = root / item
        if target.exists():
            hits.append(str(target))
    return hits


def _detect_layout_hints(root: Path) -> dict[str, list[str]]:
    # Shallow, conservative: do not speculate beyond "these paths exist".
    candidates = [
        "src",
        "include",
        "python",
        "cpp",
        "csrc",
        "torch",
        "aten",
        "c10",
        "test",
        "tests",
        "benchmarks",
        "examples",
        "scripts",
        "third_party",
        "vendor",
        "docs",
    ]
    present = [str(root / d) for d in candidates if (root / d).exists()]
    return {"present_dirs": present}


def _parse_pyproject_minimal(pyproject_path: Path) -> dict[str, object]:
    # Zero dependencies: extremely conservative extraction.
    # We only try to detect "tool.*" tables and common build-system fields.
    try:
        txt = pyproject_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}

    out: dict[str, object] = {}
    tool_tables = set(re.findall(r"(?m)^\\[tool\\.(.+?)\\]$", txt))
    if tool_tables:
        out["pyproject_tool_tables"] = sorted(tool_tables)

    m = re.search(r"(?ms)^\\[build-system\\].*?$", txt)
    if m:
        out["has_build_system_table"] = True

    return out


def _iter_rel_parts(root: Path, abs_paths: Iterable[Path]) -> Iterator[tuple[str, tuple[str, ...]]]:
    for p in abs_paths:
        try:
            rel = p.resolve().relative_to(root)
        except Exception:
            continue
        if rel.parts:
            yield str(rel), tuple(rel.parts)


def build_tree_lines(
    root: Path,
    files_for_tree: list[Path],
    *,
    depth: int,
    max_entries_per_dir: int,
    max_total_nodes: int,
) -> dict[str, object]:
    # Build a bounded, low-noise tree from the *same* file set used for stats.
    # This avoids including caches/artifacts that were pruned earlier.
    if depth <= 0:
        return {"enabled": False}

    # dir key is a tuple of parts; root is ().
    child_dirs: dict[tuple[str, ...], set[str]] = defaultdict(set)
    child_files: dict[tuple[str, ...], list[str]] = defaultdict(list)
    desc_files: Counter[tuple[str, ...]] = Counter()

    # Hard cap on ingestion to keep memory stable for huge repos.
    # This is only for tree rendering; it does not affect overall stats.
    ingest_cap = max_total_nodes * 50
    ingested = 0

    for _, parts in _iter_rel_parts(root, files_for_tree):
        ingested += 1
        if ingested > ingest_cap:
            break

        # update descendant file counts for all parent directories (including root)
        for i in range(0, len(parts)):
            desc_files[parts[:i]] += 1

        if len(parts) == 1:
            child_files[()].append(parts[0])
            continue

        parent = ()
        for i, name in enumerate(parts[:-1]):
            child_dirs[parent].add(name)
            parent = (*parent, name)
            if i == len(parts) - 2:
                child_files[parent].append(parts[-1])

    def dir_sort_key(parent: tuple[str, ...], name: str) -> tuple[int, int, str]:
        child = (*parent, name)
        priority = 0 if name in PRIORITY_DIRS else 1
        # larger descendant file counts first
        return (priority, -int(desc_files.get(child, 0)), name)

    def pick_dirs(parent: tuple[str, ...]) -> tuple[list[str], int]:
        dirs = sorted(child_dirs.get(parent, set()), key=lambda n: dir_sort_key(parent, n))
        if len(dirs) <= max_entries_per_dir:
            return dirs, 0
        keep = dirs[:max_entries_per_dir]
        omitted = len(dirs) - len(keep)
        return keep, omitted

    def pick_files(parent: tuple[str, ...]) -> tuple[list[str], int]:
        files = child_files.get(parent, [])
        if not files:
            return [], 0

        # Prefer interesting filenames; keep output bounded and stable.
        interesting = sorted({f for f in files if f in INTERESTING_FILENAMES})
        if desc_files.get(parent, 0) <= 50:
            # small directories: show up to budget, interesting first
            rest = sorted({f for f in files if f not in INTERESTING_FILENAMES})
            merged = interesting + rest
        else:
            # big directories: only show interesting
            merged = interesting

        merged = list(dict.fromkeys(merged))  # stable de-dupe
        if len(merged) <= max_entries_per_dir:
            # We still count omitted as "remaining files not shown"
            omitted = max(0, len(set(files)) - len(merged))
            return merged, omitted
        keep = merged[:max_entries_per_dir]
        omitted = max(0, len(set(files)) - len(keep))
        return keep, omitted

    lines: list[str] = []
    node_budget = max_total_nodes

    def emit(line: str) -> None:
        nonlocal node_budget
        if node_budget <= 0:
            return
        lines.append(line)
        node_budget -= 1

    def render_dir(parent: tuple[str, ...], prefix: str, level: int) -> None:
        if node_budget <= 0:
            return
        if level >= depth:
            return

        dirs, dirs_omitted = pick_dirs(parent)
        files, files_omitted = pick_files(parent)

        entries: list[tuple[str, str]] = [("dir", d) for d in dirs] + [("file", f) for f in files]
        if dirs_omitted or files_omitted:
            more = []
            if dirs_omitted:
                more.append(f"{dirs_omitted} dirs")
            if files_omitted:
                more.append(f"{files_omitted} files")
            entries.append(("meta", f"… ({', '.join(more)} omitted)"))
        total_entries = len(entries)

        for idx, (kind, name) in enumerate(entries):
            last = idx == total_entries - 1
            elbow = "└── " if last else "├── "
            if kind == "dir":
                emit(f"{prefix}{elbow}{name}/")
                child = (*parent, name)
                extension = "    " if last else "│   "
                render_dir(child, prefix + extension, level + 1)
            else:
                emit(f"{prefix}{elbow}{name}")

    emit(".")
    render_dir((), "", 0)

    return {
        "enabled": True,
        "depth": depth,
        "max_entries_per_dir": max_entries_per_dir,
        "max_total_nodes": max_total_nodes,
        "nodes_emitted": len(lines),
        "lines": lines,
        "truncated": node_budget <= 0,
    }


def build_repo_facts(
    root: Path,
    *,
    max_files: int,
    include_tree: bool,
    tree_depth: int,
    tree_max_entries: int,
    tree_max_nodes: int,
) -> dict[str, object]:
    root = root.resolve()
    git = detect_git(root)
    scan_root = Path(git.root).resolve() if git.is_repo and git.root else root

    files: Iterable[Path]
    used_git_listing = False
    files_for_tree: list[Path] = []
    if git.is_repo and git.root:
        try:
            files = iter_files_git(
                scan_root,
                rel_from=root,
                prune_dirs=set(DEFAULT_PRUNE_DIRS),
                prune_globs=set(DEFAULT_PRUNE_GLOBS),
                max_files=max_files,
            )
            used_git_listing = True
        except Exception:
            files = iter_files_non_git(
                root,
                prune_dirs=set(DEFAULT_PRUNE_DIRS),
                prune_globs=set(DEFAULT_PRUNE_GLOBS),
                max_files=max_files,
            )
    else:
        files = iter_files_non_git(
            root,
            prune_dirs=set(DEFAULT_PRUNE_DIRS),
            prune_globs=set(DEFAULT_PRUNE_GLOBS),
            max_files=max_files,
        )

    lang_counts = Counter()
    lang_bytes = defaultdict(int)
    total_files = 0
    total_bytes = 0
    largest: list[tuple[int, str]] = []

    for p in files:
        if include_tree:
            files_for_tree.append(p)
        st = _safe_stat(p)
        if st is None or not p.is_file():
            continue
        size = int(st.st_size)
        lang = _language_for_path(p)
        lang_counts[lang] += 1
        lang_bytes[lang] += size
        total_files += 1
        total_bytes += size
        if size > 0:
            largest.append((size, str(p)))

    largest.sort(reverse=True)
    largest = largest[:10]

    signals = _find_signal_paths(root)
    layout = _detect_layout_hints(root)

    pyproject_info = {}
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        pyproject_info = _parse_pyproject_minimal(pyproject)

    tree_info = {"enabled": False}
    if include_tree:
        tree_info = build_tree_lines(
            root,
            files_for_tree,
            depth=tree_depth,
            max_entries_per_dir=tree_max_entries,
            max_total_nodes=tree_max_nodes,
        )

    facts = {
        "timestamp": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "root": str(root),
        "git": dataclasses.asdict(git),
        "scan": {
            "used_git_listing": used_git_listing,
            "max_files": max_files,
            "files_counted": total_files,
            "total_bytes": total_bytes,
        },
        "languages": {
            "by_files": lang_counts.most_common(15),
            "by_bytes": sorted(lang_bytes.items(), key=lambda kv: kv[1], reverse=True)[:15],
        },
        "signals": signals,
        "layout_hints": layout,
        "pyproject": pyproject_info,
        "largest_files": largest,
        "tree": tree_info,
    }
    return facts


def _fmt_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            if u == "B":
                return f"{int(x)} {u}"
            return f"{x:.1f} {u}"
        x /= 1024.0
    return f"{n} B"


def print_markdown(facts: dict[str, object]) -> None:
    git = facts["git"]
    scan = facts["scan"]
    langs = facts["languages"]
    signals = facts["signals"]
    layout = facts["layout_hints"]
    pyproject = facts.get("pyproject") or {}
    largest = facts["largest_files"]

    def _kv(k: str, v: object) -> str:
        return f"- {k}: `{v}`"

    print("# repo_facts")
    print(_kv("root", facts["root"]))
    print(_kv("git.is_repo", git["is_repo"]))
    if git["is_repo"]:
        if git.get("root"):
            print(_kv("git.root", git["root"]))
        if git.get("branch") is not None:
            print(_kv("git.branch", git["branch"]))
        if git.get("head"):
            print(_kv("git.head", git["head"]))
        if git.get("dirty") is not None:
            print(_kv("git.dirty", git["dirty"]))

    print(_kv("scan.used_git_listing", scan["used_git_listing"]))
    print(_kv("scan.files_counted", scan["files_counted"]))
    print(_kv("scan.total_bytes", _fmt_bytes(int(scan["total_bytes"]))))

    print("\n## Languages (by files)")
    for lang, cnt in langs["by_files"][:10]:
        print(f"- `{lang}`: {cnt}")

    print("\n## Languages (by bytes)")
    for lang, b in langs["by_bytes"][:10]:
        print(f"- `{lang}`: {_fmt_bytes(int(b))}")

    if signals:
        print("\n## Signals")
        for p in signals:
            print(f"- `{p}`")

    present_dirs = layout.get("present_dirs") or []
    if present_dirs:
        print("\n## Layout hints (present dirs)")
        for d in present_dirs:
            print(f"- `{d}`")

    tool_tables = pyproject.get("pyproject_tool_tables") or []
    if tool_tables:
        print("\n## pyproject.toml (tool tables)")
        for t in tool_tables[:25]:
            print(f"- `tool.{t}`")

    if largest:
        print("\n## Largest files (top 10)")
        for size, p in largest:
            print(f"- {_fmt_bytes(int(size))}: `{p}`")

    tree = facts.get("tree") or {"enabled": False}
    if tree.get("enabled"):
        depth = tree.get("depth", "?")
        truncated = tree.get("truncated", False)
        suffix = " (truncated)" if truncated else ""
        print(f"\n## Tree (depth {depth}){suffix}")
        print("```")
        for line in tree.get("lines", []):
            print(line)
        print("```")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Summarize a repo into a compact facts packet.")
    ap.add_argument("path", nargs="?", default=".", help="Repo root (or any directory within it).")
    ap.add_argument("--json", action="store_true", help="Output JSON instead of Markdown.")
    ap.add_argument("--max-files", type=int, default=200_000, help="Cap scanned file count.")
    ap.add_argument("--tree", action="store_true", help="Include a bounded tree view.")
    ap.add_argument("--tree-depth", type=int, default=3, help="Tree depth (default: 3).")
    ap.add_argument(
        "--tree-max-entries",
        type=int,
        default=30,
        help="Max dirs/files shown per directory (default: 30).",
    )
    ap.add_argument(
        "--tree-max-nodes",
        type=int,
        default=1200,
        help="Max total emitted tree lines (default: 1200).",
    )
    args = ap.parse_args(argv)

    root = Path(args.path)
    if not root.exists():
        print(f"error: path does not exist: {root}", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"error: path is not a directory: {root}", file=sys.stderr)
        return 2

    if args.max_files <= 0:
        print("error: --max-files must be > 0", file=sys.stderr)
        return 2

    if args.tree_depth <= 0:
        print("error: --tree-depth must be > 0", file=sys.stderr)
        return 2
    if args.tree_max_entries <= 0:
        print("error: --tree-max-entries must be > 0", file=sys.stderr)
        return 2
    if args.tree_max_nodes <= 0:
        print("error: --tree-max-nodes must be > 0", file=sys.stderr)
        return 2

    facts = build_repo_facts(
        root,
        max_files=args.max_files,
        include_tree=bool(args.tree),
        tree_depth=int(args.tree_depth),
        tree_max_entries=int(args.tree_max_entries),
        tree_max_nodes=int(args.tree_max_nodes),
    )
    if args.json:
        print(json.dumps(facts, ensure_ascii=False, indent=2))
    else:
        print_markdown(facts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
