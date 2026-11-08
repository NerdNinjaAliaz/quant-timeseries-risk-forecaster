#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="quant-timeseries-risk-forecaster"
U1="NerdNinja"
U2="Aliaz"
GITHUB_USER="${U1}${U2}"
REPO_DESC="Deep Learning Financial Risk and Portfolio Backtesting Engine (PyTorch Stacked LSTM/GRU vs. Quantitative Baselines)"
REMOTE_URL="https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

echo "1. Preparing local Git repository for ${GITHUB_USER}/${REPO_NAME}..."

if [ ! -d ".git" ]; then
  git init -b main
fi

git config user.name "Ali Azizi"
git config user.email "aziziali.9473@gmail.com"

git add .
if ! git diff --cached --quiet; then
  git commit -m "feat: initial release of quant-timeseries-risk-forecaster (PyTorch LSTM/GRU risk engine)"
fi

if git remote | grep -q "^origin$"; then
  git remote set-url origin "${REMOTE_URL}"
else
  git remote add origin "${REMOTE_URL}"
fi

echo "2. Checking or creating remote repository on GitHub (${GITHUB_USER}/${REPO_NAME})..."

REPO_CREATED=0

# Method A: GitHub CLI if installed and logged in
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  echo "   -> Creating repository via GitHub CLI..."
  gh repo create "${GITHUB_USER}/${REPO_NAME}" --public --description "${REPO_DESC}" --source=. --remote=origin --push && exit 0
fi

# Method B: Use token from GITHUB_TOKEN env var or macOS git-credential-osxkeychain
TOKEN="${GITHUB_TOKEN:-}"
if [ -z "${TOKEN}" ]; then
  CRED_OUT="$(printf "protocol=https\nhost=github.com\n\n" | git credential fill 2>/dev/null || true)"
  TOKEN="$(printf "%s\n" "${CRED_OUT}" | sed -n 's/^password=//p' | head -n 1)"
fi

if [ -n "${TOKEN}" ]; then
  HTTP_CODE="$(curl -sS -o /tmp/gh_repo_create.json -w "%{http_code}" \
    -X POST "https://api.github.com/user/repos" \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${TOKEN}" \
    -d "{\"name\":\"${REPO_NAME}\",\"description\":\"${REPO_DESC}\",\"private\":false}" || true)"

  if [ "${HTTP_CODE}" = "201" ] || [ "${HTTP_CODE}" = "422" ]; then
    echo "   -> GitHub repository ${GITHUB_USER}/${REPO_NAME} verified/created via GitHub API (HTTP ${HTTP_CODE})."
    REPO_CREATED=1
  else
    echo "   -> Cached Git credential could not auto-create the repository via API (HTTP ${HTTP_CODE})."
  fi
fi

# Method C: Check if the public repo already exists on GitHub
if [ "${REPO_CREATED}" -eq 0 ]; then
  CHECK_CODE="$(curl -sS -o /dev/null -w "%{http_code}" "https://github.com/${GITHUB_USER}/${REPO_NAME}" || true)"
  if [ "${CHECK_CODE}" = "200" ]; then
    REPO_CREATED=1
  fi
fi

# Method D: Open pre-filled GitHub New Repository page in macOS browser if still not created
if [ "${REPO_CREATED}" -eq 0 ]; then
  NEW_REPO_URL="https://github.com/new?name=${REPO_NAME}&description=Deep+Learning+Financial+Risk+and+Portfolio+Backtesting+Engine+%28PyTorch+LSTM%2FGRU%29"
  echo ""
  echo "================================================================================"
  echo "Git cannot push until the empty '${REPO_NAME}' repository exists on GitHub."
  echo "Opening pre-filled GitHub repository creation page in your browser..."
  echo "  -> Make sure Owner is '${GITHUB_USER}' and leave 'Add a README file' UNCHECKED."
  echo "  -> Click the green 'Create repository' button in your browser."
  echo "================================================================================"
  open "${NEW_REPO_URL}" 2>/dev/null || true
  read -r -p "Press [Enter] here in Terminal as soon as you click 'Create repository'..." _
fi

echo "3. Pushing branch 'main' to ${REMOTE_URL}..."
git push -u origin main

echo ""
echo "Success! Your repository is live at: https://github.com/${GITHUB_USER}/${REPO_NAME}"
