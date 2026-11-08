#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="quant-timeseries-risk-forecaster"
U1="NerdNinja"
U2="Aliaz"
GITHUB_USER="${U1}${U2}"
REPO_DESC="Deep Learning Financial Risk & Portfolio Backtesting Engine (PyTorch Stacked LSTM/GRU vs. Quantitative Baselines — Sharpe, Sortino, VaR, CVaR & Streamlit Dashboard)"

echo "=== Step 1: Pushing latest test suite & CI updates to ${GITHUB_USER}/${REPO_NAME} ==="
git add .
if ! git diff --cached --quiet; then
  git commit -m "test: add automated functional verification suite and profile SEO automation"
fi
git push origin main

# Retrieve GitHub token from env or macOS Keychain
TOKEN="${GITHUB_TOKEN:-}"
if [ -z "${TOKEN}" ]; then
  CRED_OUT="$(printf "protocol=https\nhost=github.com\n\n" | git credential fill 2>/dev/null || true)"
  TOKEN="$(printf "%s\n" "${CRED_OUT}" | sed -n 's/^password=//p' | head -n 1)"
fi

if [ -n "${TOKEN}" ]; then
  echo ""
  echo "=== Step 2: Applying 14 Search-Optimized GitHub Topics & Metadata via API ==="
  curl -sS -X PATCH "https://api.github.com/repos/${GITHUB_USER}/${REPO_NAME}" \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${TOKEN}" \
    -d "{\"description\":\"${REPO_DESC}\",\"has_issues\":true,\"has_projects\":true,\"has_wiki\":true}" >/dev/null || true

  TOPICS_HTTP="$(curl -sS -o /tmp/gh_topics.json -w "%{http_code}" \
    -X PUT "https://api.github.com/repos/${GITHUB_USER}/${REPO_NAME}/topics" \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${TOKEN}" \
    -d '{"names":["pytorch","quantitative-finance","time-series-forecasting","lstm","gru","risk-management","portfolio-optimization","backtesting","value-at-risk","sharpe-ratio","streamlit","bitcoin-price-prediction","deep-learning","algorithmic-trading"]}' || true)"
  echo "   -> Repository topics API status: HTTP ${TOPICS_HTTP}"

  echo ""
  echo "=== Step 3: Creating/Updating Special Profile Repo (${GITHUB_USER}/${GITHUB_USER}) ==="
  curl -sS -X POST "https://api.github.com/user/repos" \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${TOKEN}" \
    -d "{\"name\":\"${GITHUB_USER}\",\"description\":\"Ali Azizi — AI Product Manager & Biomedical AI Engineer Portfolio\",\"private\":false}" >/dev/null || true
fi

PROFILE_DIR="/tmp/${GITHUB_USER}-profile-repo"
rm -rf "${PROFILE_DIR}"
mkdir -p "${PROFILE_DIR}"
cp docs/GITHUB_PROFILE_README.md "${PROFILE_DIR}/README.md"
cd "${PROFILE_DIR}"
git init -b main
git config user.name "Ali Azizi"
git config user.email "aziziali.9473@gmail.com"
git add README.md
git commit -m "docs: publish executive AI & Business portfolio profile README"
git remote add origin "https://github.com/${GITHUB_USER}/${GITHUB_USER}.git"

echo "Pushing Profile README to https://github.com/${GITHUB_USER}/${GITHUB_USER}.git ..."
if ! git push -u origin main --force 2>/dev/null; then
  echo "Opening GitHub new repo page for your profile README (${GITHUB_USER}/${GITHUB_USER})..."
  open "https://github.com/new?name=${GITHUB_USER}&description=Ali+Azizi+AI+Product+Manager+Portfolio" 2>/dev/null || true
  read -r -p "Click 'Create repository' in your browser (leave 'Add a README' unchecked), then press [Enter]..." _
  git push -u origin main --force
fi

echo ""
echo "=== All Done! ==="
echo "1. Repository + 14 Search Topics: https://github.com/${GITHUB_USER}/${REPO_NAME}"
echo "2. Profile Landing Page:          https://github.com/${GITHUB_USER}"
