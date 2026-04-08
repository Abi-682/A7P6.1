#!/usr/bin/env bash
set -euo pipefail

PY_VERSION="3.13"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed."
  echo "Install uv first:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

echo "Installing Python ${PY_VERSION}..."
uv python install "${PY_VERSION}"

echo "Creating virtual environment and installing dependencies..."
uv sync

echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Open in VS Code: code ."
echo "  2. Update README.md with your project description"
echo "  3. Start coding in src/"
