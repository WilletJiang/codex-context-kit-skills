#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CODEX_HOME="${CODEX_HOME:-${HOME}/.codex}"
SKILLS_DST="${CODEX_HOME}/skills"
BIN_DST="${CODEX_HOME}/bin"

SKILLS=(ctx def sym repo_facts)
BINS=(ctx def sym repo_facts rf)

mkdir -p "$SKILLS_DST" "$BIN_DST"

for s in "${SKILLS[@]}"; do
  rm -rf "${SKILLS_DST:?}/${s}"
  cp -R "${REPO_ROOT}/skills/${s}" "${SKILLS_DST}/"
done

for b in "${BINS[@]}"; do
  install -m 0755 "${REPO_ROOT}/bin/${b}" "${BIN_DST}/${b}"
done

cat <<EOF
Installed:
- skills -> ${SKILLS_DST}/{${SKILLS[*]}}
- wrappers -> ${BIN_DST}/{${BINS[*]}}

Enable commands in new shells by adding this to your shell startup:
  export PATH="${BIN_DST}:\$PATH"

Current shell (zsh/bash):
  export PATH="${BIN_DST}:\$PATH"
EOF
