#!/bin/bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: arxiv-download.sh <arxiv-id> <papers-dir>"
  echo "Example: arxiv-download.sh 2401.04088 workspace/papers"
  exit 1
fi

ARXIV_ID="$1"
PAPERS_DIR="$2"

# Validate arXiv ID format
if ! [[ "$ARXIV_ID" =~ ^[0-9]{4}\.[0-9]{4,5}(v[0-9]+)?$ ]]; then
  echo "Error: Invalid arXiv ID format: $ARXIV_ID"
  echo "Expected format: YYMM.NNNNN (e.g., 2401.04088)"
  exit 1
fi

# 1. Fetch title from API
echo "Fetching metadata for $ARXIV_ID..."
XML=$(curl -sL --connect-timeout 10 --max-time 30 "https://export.arxiv.org/api/query?id_list=$ARXIV_ID")
TITLE=$(echo "$XML" | sed -n 's/.*<title>\(.*\)<\/title>.*/\1/p' | tail -1)

if [ -z "$TITLE" ]; then
  echo "Error: Could not fetch title for $ARXIV_ID"
  exit 1
fi

# 2. Create folder
SLUG=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g' | sed 's/__*/_/g' | sed 's/_$//')
DEST="$PAPERS_DIR/${ARXIV_ID}_${SLUG}"

if [ -d "$DEST" ]; then
  echo "Already exists: $DEST (skipping download)"
  exit 0
fi

mkdir -p "$DEST"
echo "Title: $TITLE"
echo "Folder: $DEST"

# Set up cleanup trap for temp files
TMP=""
cleanup() { [ -n "$TMP" ] && rm -f "$TMP"; }
trap cleanup EXIT

# 3. Download BibTeX
echo "Downloading citation..."
curl -sL --connect-timeout 10 --max-time 30 "https://arxiv.org/bibtex/$ARXIV_ID" > "$DEST/citation.bib"

# 4. Download PDF
echo "Downloading PDF..."
curl -sL --connect-timeout 10 --max-time 120 -o "$DEST/${ARXIV_ID}.pdf" "https://arxiv.org/pdf/$ARXIV_ID"

# 5. Download and extract TeX source
echo "Downloading source..."
TMP=$(mktemp)
curl -sL --connect-timeout 10 --max-time 120 -o "$TMP" "https://arxiv.org/e-print/$ARXIV_ID"
mkdir -p "$DEST/source"
if ! tar xzf "$TMP" -C "$DEST/source" 2>/dev/null; then
  if file "$TMP" | grep -q gzip; then
    gunzip -c "$TMP" > "$DEST/source/main.tex"
  else
    cp "$TMP" "$DEST/source/main.tex"
  fi
fi

# 6. Convert LaTeX to markdown via pandoc (if source available)
MAIN_TEX=$(find "$DEST/source" -name "*.tex" -exec grep -l '\\begin{document}' {} \; 2>/dev/null | head -1)
if [ -n "$MAIN_TEX" ] && command -v pandoc &>/dev/null; then
  echo "Converting LaTeX to markdown..."
  TEXDIR=$(dirname "$MAIN_TEX")
  (cd "$TEXDIR" && pandoc "$(basename "$MAIN_TEX")" -t markdown -o "$DEST/paper.md" 2>/dev/null) || true
fi

echo ""
echo "✓ Downloaded to: $DEST"
echo "  PDF:      ${ARXIV_ID}.pdf"
echo "  Source:   source/"
echo "  Citation: citation.bib"
[ -f "$DEST/paper.md" ] && echo "  Markdown: paper.md (from LaTeX)"
