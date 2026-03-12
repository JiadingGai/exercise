#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-.}"

echo "ClaudePaperSurveyAgent Installer"
echo "================================"
echo ""

# Check prerequisites
MISSING=0
check_cmd() {
  if command -v "$1" &>/dev/null; then
    echo "  ✓ $1 found: $(command -v "$1")"
  else
    echo "  ✗ $1 NOT FOUND — install with: $2"
    MISSING=1
  fi
}

echo "Checking prerequisites..."
check_cmd "claude" "See https://docs.anthropic.com/en/docs/claude-code"
check_cmd "uvx" "brew install uv"
check_cmd "npx" "brew install node"
check_cmd "pandoc" "brew install pandoc"
check_cmd "pdflatex" "brew install --cask mactex-no-gui"
echo ""

if [ "$MISSING" -eq 1 ]; then
  echo "Warning: Some prerequisites are missing. Install them and re-run."
  echo "  The agent will still be installed, but may not work fully."
  echo ""
fi

# Resolve target directory
TARGET=$(cd "$TARGET" && pwd)
echo "Installing to: $TARGET"

# Copy project config
cp "$SCRIPT_DIR/CLAUDE.md" "$TARGET/"
cp "$SCRIPT_DIR/.mcp.json" "$TARGET/"
cp -r "$SCRIPT_DIR/.claude" "$TARGET/"
cp -r "$SCRIPT_DIR/scripts" "$TARGET/"
cp "$SCRIPT_DIR/.gitignore" "$TARGET/" 2>/dev/null || true

# Create workspace structure
mkdir -p "$TARGET/workspace"/{papers,notes,survey,slides}

# Copy Makefiles to workspace
cp "$SCRIPT_DIR/workspace/survey/Makefile" "$TARGET/workspace/survey/"
cp "$SCRIPT_DIR/workspace/slides/Makefile" "$TARGET/workspace/slides/"

# Make scripts executable
chmod +x "$TARGET/scripts/"*.sh

echo ""
echo "✓ ClaudePaperSurveyAgent installed to $TARGET"
echo ""
echo "Usage:"
echo "  cd $TARGET"
echo "  claude"
echo "  > Write a survey on [your topic], covering ~20 papers"
echo ""
echo "Features:"
echo "  • Agent teams for parallel paper reading"
echo "  • 4 skills: search-papers, read-paper, write-survey, make-slides"
echo "  • 5 MCP servers: document-loader, fetch, git, memory, sequentialthinking"
echo "  • LaTeX survey + Beamer slides with Makefiles"
