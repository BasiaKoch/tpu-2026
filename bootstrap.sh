#!/usr/bin/env bash
#
# Bring a fresh TPU VM up to a working state:
#   - Python 3.12 + venv at ~/venvs/tunix
#   - tunix / jax / flax stack
#   - secrets pulled from Google Secret Manager (optional)
#   - venv + .env auto-loaded in interactive shells
#
# Usage on a freshly-created VM:
#   git clone https://github.com/borisbolliet/tpu-2026.git
#   cd tpu-2026 && ./bootstrap.sh
#
# Idempotent — safe to re-run.

set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV=${VENV:-$HOME/venvs/tunix}
SECRET_NAME=${SECRET_NAME:-tunix-env}
PROJECT_ID=${PROJECT_ID:-tpu-2026}

echo "==> Installing python3.12 via uv"
# The deadsnakes PPA is preconfigured on these VMs but
# ppa.launchpadcontent.net is not reachable from the internal-IP TPU subnet
# (connection times out via Cloud NAT). uv pulls a prebuilt CPython from
# GitHub instead, which the firewall does allow.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
PYTHON312=$(uv python find 3.12)

if [[ ! -d "$VENV" ]]; then
  echo "==> Creating venv at $VENV"
  "$PYTHON312" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip setuptools wheel

echo "==> Installing pinned deps from requirements.txt"
# requirements.txt includes libtpu — without it, jax silently falls back to CPU
# on a TPU VM ("A Google TPU may be present ... Falling back to cpu").
pip install -r "$REPO_DIR/requirements.txt"

echo "==> Installing jax / tunix / qwix / flax from GitHub HEAD"
# Order matters: tunix pulls flax from PyPI, so we replace flax last.
pip install git+https://github.com/jax-ml/jax
pip install git+https://github.com/google/tunix git+https://github.com/google/qwix
pip uninstall -y flax
pip install git+https://github.com/google/flax

echo "==> Fetching secrets from Secret Manager (skipped if unavailable)"
if command -v gcloud >/dev/null && \
   gcloud secrets describe "$SECRET_NAME" --project="$PROJECT_ID" >/dev/null 2>&1; then
  gcloud secrets versions access latest \
    --secret="$SECRET_NAME" --project="$PROJECT_ID" > "$HOME/.env"
  chmod 600 "$HOME/.env"
  echo "    wrote ~/.env"
else
  echo "    secret '$SECRET_NAME' not found — create it with:"
  echo "    gcloud secrets create $SECRET_NAME --data-file=<file> --project=$PROJECT_ID"
fi

echo "==> Wiring venv + .env + uv into ~/.bashrc"
grep -qF 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null \
  || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
grep -qF "source $VENV/bin/activate" "$HOME/.bashrc" 2>/dev/null \
  || echo "source $VENV/bin/activate" >> "$HOME/.bashrc"
grep -qF "set -a; source ~/.env; set +a" "$HOME/.bashrc" 2>/dev/null \
  || echo '[ -f ~/.env ] && set -a && source ~/.env && set +a' >> "$HOME/.bashrc"

echo "==> Done. Open a new shell or:  source ~/.bashrc"
