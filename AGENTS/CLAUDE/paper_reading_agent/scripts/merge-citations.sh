#!/bin/bash
set -euo pipefail

PAPERS_DIR="${1:-workspace/papers}"
OUTPUT="${2:-workspace/survey/refs.bib}"

mkdir -p "$(dirname "$OUTPUT")"

echo "% Auto-generated bibliography — merged from downloaded papers" > "$OUTPUT"
echo "% Source: $PAPERS_DIR/*/citation.bib" >> "$OUTPUT"
echo "% Re-run: bash scripts/merge-citations.sh" >> "$OUTPUT"
echo "" >> "$OUTPUT"

COUNT=0
for bib in "$PAPERS_DIR"/*/citation.bib; do
  if [ -f "$bib" ]; then
    PAPER_DIR=$(dirname "$bib" | xargs basename)
    echo "% --- $PAPER_DIR ---" >> "$OUTPUT"
    cat "$bib" >> "$OUTPUT"
    echo "" >> "$OUTPUT"
    COUNT=$((COUNT + 1))
  fi
done

echo "✓ Merged $COUNT citations into $OUTPUT"
