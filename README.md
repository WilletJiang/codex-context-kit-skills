# codex-context-kit

Small, fast, bounded “context packets” for navigating codebases (git or non-git).
Designed to pair well with Codex CLI, but usable standalone.

## Design goals

- **Fast**: `rg` for discovery; read as little as possible
- **Bounded**: hard caps on hits/files/output (token-safe by default)
- **Low-mislead**: Python uses AST for `sym`/`def` whenever possible

## What you get (commands)

- `repo_facts`: repo overview + bounded tree (great first command)
- `sym`: Python symbol index via AST (`--query/--kind/--base/--decorator`)
- `def`: definition locator (`confidence=exact` for Python via AST; otherwise candidates)
- `ctx`: bounded context extractor around `rg` matches (default has scope headers; optional `--json`)

## Requirements

- `python3`
- `rg` (ripgrep) on PATH

## Install (macOS/Linux, bash/zsh)

```sh
git clone https://github.com/WilletJiang/codex-context-kit-skills.git
cd codex-context-kit
./install.sh
```

The installer copies into `~/.codex/{skills,bin}` by default.
To install elsewhere:

```sh
CODEX_HOME="$HOME/.local/codex" ./install.sh
```

Enable commands:

```sh
export PATH="$HOME/.codex/bin:$PATH"
```

Add it permanently:
- zsh: add to `~/.zshrc`
- bash: add to `~/.bashrc` (or `~/.profile` for login shells)

## Update

```sh
cd codex-context-kit
git pull
./install.sh
```

## Quickstart (workflow)

From a repo root:

```sh
repo_facts .
sym . --query config --kind class
def MyClass . --lang py -C 2
ctx 'PATTERN' . -F
```

Common Python filters:

```sh
sym . --base nn.Module --kind class
sym . --decorator dataclass --kind class
```

Chain: symbol -> definition -> minimal context:

```sh
sym . --query Trainer --kind class
def MyTrainer . --lang py -C 1
ctx 'trainer = MyTrainer' . -F -C 2
```

## Output conventions

- Text mode outputs include clickable locations starting with `path:line:col`.
- JSON mode is available for `ctx` and `def` (and `sym --json`) to support pipelines.

## Correctness notes

- `sym` and Python `def` results are AST-derived (low false positives).
- Non-Python `def` results are candidates; confirm with `ctx` before editing.
- `repo_facts` is an inventory of files present; it does not prove how a project builds/runs.

## Uninstall

Remove installed files (adjust if you changed `CODEX_HOME`):

```sh
rm -rf "$HOME/.codex/skills/ctx" "$HOME/.codex/skills/def" "$HOME/.codex/skills/sym" "$HOME/.codex/skills/repo_facts"
rm -f "$HOME/.codex/bin/ctx" "$HOME/.codex/bin/def" "$HOME/.codex/bin/sym" "$HOME/.codex/bin/repo_facts" "$HOME/.codex/bin/rf"
```

## License

MIT. See `LICENSE`.
