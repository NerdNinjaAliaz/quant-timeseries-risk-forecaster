#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="quant-timeseries-risk-forecaster"
GITHUB_USER="NerdNinjaAliaz"

echo "Preparing local Git repository for ${GITHUB_USER}/${REPO_NAME}..."

if [ ! -d ".git" ]; then
  git init -b main
fi

git config user.name "Ali Azizi"
git config user.email "aziziali.9473@gmail.com"

git add .
if ! git diff --cached --quiet; then
  git commit -m "feat: initial release of quant-timeseries-risk-forecaster (PyTorch LSTM/GRU risk engine)"
fi

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  echo "GitHub CLI detected and authenticated. Creating and pushing ${GITHUB_USER}/${REPO_NAME}..."
  gh repo create "${GITHUB_USER}/${REPO_NAME}" --public --source=. --remote=origin --push
else
  echo "Pushing branch 'main' to remote 'origin'..."
  git push -u origin main
fi
