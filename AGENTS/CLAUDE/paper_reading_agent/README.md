# ClaudePaperSurveyAgent

An autonomous survey paper writing system powered by Claude Code agent teams.

Give it a research topic → it discovers papers, reads them in parallel, builds a taxonomy, writes a LaTeX survey, and generates Beamer presentation slides.

This repository is best understood as a project template for Claude Code rather than a standalone application server.
It installs a survey-writing workspace containing:

- a team-lead playbook in `CLAUDE.md`
- MCP server configuration in `.mcp.json`
- reusable skills for search, reading, writing, and slide generation
- helper scripts for downloading arXiv papers and merging citations
- Makefiles for compiling the final survey paper and slides

The intended workflow is: install this template into a fresh project folder, open that folder in Claude Code, then ask Claude to produce a survey on a topic you care about.

## Quick Start

```bash
# Clone and install into a new survey project
git clone <repo-url> ClaudePaperSurveyAgent
mkdir -p ~/Documents/surveys/my-survey
bash ClaudePaperSurveyAgent/install.sh ~/Documents/surveys/my-survey

# Start the agent
cd ~/Documents/surveys/my-survey
claude

# Tell it what to survey
> Write a survey on flash attention use in RL training efficiency.
> Cover ~20 papers from 2022-2025. Target: IEEE format, 15 pages.
```

## Prerequisites

- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (v2.1+)
- `uv` — `brew install uv` (for document-loader MCP server)
- `node` — `brew install node` (for fetch/git/memory/sequentialthinking MCP servers)
- `pandoc` — `brew install pandoc` (for LaTeX → markdown conversion)
- `pdflatex` — `brew install --cask mactex-no-gui` (for compiling LaTeX)

You also need working network access during survey generation because the workflow queries arXiv, Semantic Scholar, and MCP-backed services.

## What This Repo Does

At a high level, the project turns Claude Code into a small research team:

- a **Researcher** agent searches for papers, downloads sources, and collects citations
- multiple **Reader** agents read papers in parallel and write structured notes
- a **Writer** agent builds the taxonomy, drafts the survey paper, generates slides, and compiles outputs
- the **Team Lead** coordinates the workflow and pauses for human approval at important checkpoints

The repo does not ship model weights, a web UI, or a background daemon.
Instead, it provides instructions, scripts, and templates that Claude follows inside your workspace.

## What Gets Installed

Running `install.sh` copies the project scaffold into a target survey directory:

- `CLAUDE.md` — the top-level operating instructions for the team lead
- `.mcp.json` — MCP servers for document loading, fetch, git, memory, and sequential thinking
- `.claude/settings.json` — enables agent team support
- `.claude/skills/` — the reusable workflows for each stage of the pipeline
- `scripts/arxiv-download.sh` — download PDF, source, and BibTeX for an arXiv paper
- `scripts/merge-citations.sh` — merge downloaded BibTeX files into `refs.bib`
- `workspace/` — empty working directories plus Makefiles for survey and slides

The installer does not pre-download papers or run the pipeline automatically.
It only prepares the workspace so Claude can run the workflow later.

## Installation

The installer is a plain shell script:

```bash
bash install.sh /path/to/my-survey
```

Internally it:

- checks whether `claude`, `uvx`, `npx`, `pandoc`, and `pdflatex` are available
- copies the repo's configuration files and scripts into the target directory
- creates `workspace/papers`, `workspace/notes`, `workspace/survey`, and `workspace/slides`
- marks the helper shell scripts as executable

After installation, your survey project becomes a normal Claude Code workspace.

## How To Use It

1. Open the installed survey project in Claude Code.
2. Start with a concrete prompt describing your topic, scope, time window, and output format.
3. Review the generated paper shortlist when Claude pauses at the first checkpoint.
4. Approve or refine the taxonomy at the second checkpoint.
5. Review the compiled paper and slide deck at the final checkpoint.

Typical prompts look like:

```text
Write a survey on inference-time alignment methods for large language models.
Cover roughly 25 papers from 2022-2026.
Target IEEE format and generate a 15-20 slide Beamer deck.
```

Useful parameters to specify up front:

- topic or subfield
- approximate paper count
- year range
- target format such as IEEE or ACM
- desired paper length or slide count
- cost preferences such as "use Sonnet for readers"

## Manual Commands

If you want to inspect or run parts of the workflow yourself, these are the main entrypoints:

```bash
# Download a single paper from arXiv
bash scripts/arxiv-download.sh 2401.04088 workspace/papers

# Merge all downloaded BibTeX files into the survey bibliography
bash scripts/merge-citations.sh

# Compile the survey and slides
make -C workspace/survey
make -C workspace/slides
```

## How It Works

### Architecture

```
You → Claude Code (Team Lead)
       ├── Researcher      — searches arXiv + Semantic Scholar, downloads papers
       ├── Reader-1 ─┐
       ├── Reader-2  ├──── read papers in parallel, extract structured notes
       ├── Reader-3 ─┘
       └── Writer          — synthesizes taxonomy, writes LaTeX survey + Beamer slides
```

### Pipeline

1. **You** provide a topic and parameters (paper count, years, format)
2. **Researcher** searches arXiv + Semantic Scholar APIs, discovers seed papers, expands via citation graph
3. **⏸ Checkpoint:** Team lead presents paper list → you approve
4. **Researcher** downloads approved papers (PDF + LaTeX source + BibTeX)
5. **Readers** (3 in parallel) read papers, extract structured notes, store findings in memory
6. **Writer** reads all notes, builds taxonomy using sequential thinking
7. **⏸ Checkpoint:** Team lead presents taxonomy → you approve
8. **Writer** writes LaTeX survey + Beamer slides, compiles via `make`
9. **⏸ Checkpoint:** Team lead presents final output → you review

### Output

```
workspace/
├── paper-list.md         ← shortlisted papers for approval
├── taxonomy.md           ← writer-produced taxonomy for approval
├── papers/               ← downloaded PDFs, LaTeX sources, BibTeX
├── notes/                ← structured notes written by reader agents
├── survey/
│   ├── main.tex        ← LaTeX survey paper
│   ├── refs.bib        ← Merged bibliography
│   ├── main.pdf        ← Compiled PDF
│   └── Makefile
└── slides/
    ├── slides.tex      ← Beamer presentation
    ├── refs.bib
    ├── slides.pdf      ← Compiled PDF
    └── Makefile
```

Build manually:
```bash
make -C workspace/survey    # compile survey
make -C workspace/slides    # compile slides
```

## Features

### Claude Code Agent Teams
- **5 teammates** working in parallel with a shared task list
- Task dependencies ensure correct ordering
- Teammates communicate via messaging and shared memory
- Human checkpoints at paper selection and taxonomy approval

### Skills (4)
| Skill | Purpose |
|-------|---------|
| `/search-papers` | arXiv + Semantic Scholar search, citation graph expansion |
| `/read-paper` | Structured paper reading with template-based notes |
| `/write-survey` | LaTeX survey writing with taxonomy-driven structure |
| `/make-slides` | Beamer slide generation from survey content |

### MCP Servers (5)
| Server | Source | Purpose |
|--------|--------|---------|
| `document-loader` | AWS Labs | Parse PDFs with equation preservation |
| `fetch` | Anthropic | HTTP requests to arXiv/Semantic Scholar APIs |
| `git` | Anthropic | Version control, push to Overleaf |
| `memory` | Anthropic | Persistent knowledge graph across teammates |
| `sequentialthinking` | Anthropic | Structured reasoning for taxonomy building |

### Scientific Integrity
- Citations sourced only from arXiv/Semantic Scholar BibTeX (never fabricated)
- Uncertain claims marked with `[NEEDS VERIFICATION]`
- Missing data uses placeholders (`XX.X%`)
- Human approval required at key checkpoints

## Checkpoints and Human Review

This project is intentionally not "one prompt and forget it."
The workflow pauses for user approval at three moments:

1. paper shortlist approval
2. taxonomy approval
3. final paper and slide review

That review loop is part of the design.
It reduces fabricated coverage, keeps the survey in scope, and gives you a chance to steer the project before the expensive writing stages.

## Repository Roles

The most important files in the repo are:

- `CLAUDE.md` — defines the team-lead behavior, required checkpoints, and teammate responsibilities
- `.mcp.json` — tells Claude Code which MCP servers to launch
- `.claude/skills/search-papers/SKILL.md` — paper discovery workflow
- `.claude/skills/read-paper/SKILL.md` — structured reading workflow and note template usage
- `.claude/skills/write-survey/SKILL.md` — taxonomy and paper drafting workflow
- `.claude/skills/make-slides/SKILL.md` — slide generation workflow
- `scripts/` — shell helpers for download and citation merge
- `workspace/*/Makefile` — LaTeX build entrypoints

If you want to change the behavior of the system, these are the files you will edit most often.

## Customization

### Change survey format
Edit the LaTeX template: `.claude/skills/write-survey/templates/survey.tex`

The default uses IEEE format. Replace with ACM, NeurIPS, or any other class.

### Change slide format
Edit: `.claude/skills/make-slides/templates/beamer.tex`

### Adjust paper reading depth
Edit: `.claude/skills/read-paper/SKILL.md` and `template.md`

### Cost savings
Tell the team lead: "Use Sonnet for readers" — reduces cost from ~$26 to ~$12 per survey.

## Cost Estimate

| Configuration | Estimated Cost |
|--------------|---------------|
| All Opus 4.6 (default) | ~$26 |
| Sonnet for readers | ~$12 |
| Writer revision only | ~$8 |

## Current Limitations

This project is a workflow scaffold, not a hardened production system.
Some practical caveats:

- it depends on external services such as arXiv, Semantic Scholar, and MCP-backed tools being available
- paper discovery and download quality still depends on the quality of the search prompt and upstream metadata
- generated LaTeX should still be reviewed by a human before submission or publication
- bibliography and build artifacts are only as good as the downloaded sources and extracted citation data

In practice, it works best when you treat Claude as a fast collaborator rather than a fully unattended paper factory.

## Project Structure

```
ClaudePaperSurveyAgent/
├── CLAUDE.md                              ← Team lead playbook
├── .mcp.json                              ← 5 MCP servers
├── .claude/
│   ├── settings.json                      ← Agent teams enabled
│   └── skills/
│       ├── search-papers/SKILL.md
│       ├── read-paper/
│       │   ├── SKILL.md
│       │   └── template.md
│       ├── write-survey/
│       │   ├── SKILL.md
│       │   └── templates/survey.tex
│       └── make-slides/
│           ├── SKILL.md
│           └── templates/beamer.tex
├── scripts/
│   ├── arxiv-download.sh
│   └── merge-citations.sh
├── workspace/
│   ├── survey/Makefile
│   └── slides/Makefile
├── install.sh
├── .gitignore
└── README.md
```

## License

MIT
