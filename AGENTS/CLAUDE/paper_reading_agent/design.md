# ClaudePaperSurveyAgent Design

## Overview

ClaudePaperSurveyAgent is a Claude Code project template for producing a survey paper and a matching slide deck from a user-provided research topic.

The system is instruction-driven rather than application-driven:

- `CLAUDE.md` defines the top-level operating model
- `.claude/skills/` contains reusable task-specific workflows
- `.mcp.json` configures the external MCP tools the workflow depends on
- `scripts/` contains lightweight shell helpers for paper download and citation merging
- `workspace/` is the working area where papers, notes, LaTeX, and slides are generated

The repo does not provide a standalone backend service or web UI.
Instead, it scaffolds a Claude Code workspace that behaves like a small agent team.

## Design Goals

- Discover and read full papers rather than relying on abstracts alone
- Split the workload across specialized agents
- Preserve structured human checkpoints for approval
- Produce editable LaTeX artifacts rather than opaque final-only outputs
- Reuse shared memory and repeatable skills across the pipeline

## Non-Goals

- Fully unattended publication-quality paper writing
- Support for arbitrary paper sources beyond the configured workflow
- A hardened production-grade downloader or crawler
- A general literature management system

## Agent Team Model

The system is organized around one coordinator and four worker roles.

### Team Lead

The Team Lead is the Claude session controlled by the user.
Its responsibilities are:

- interpret the survey request
- create the agent team
- assign tasks and dependencies
- enforce human checkpoints
- verify tool availability before the workflow starts
- present final outputs

The Team Lead behavior is defined in `CLAUDE.md`.

### Researcher

The Researcher owns discovery and acquisition.
Its responsibilities are:

- search arXiv and Semantic Scholar
- identify seed papers
- expand via citation graph
- rank and filter candidate papers
- download approved papers and metadata
- generate `workspace/paper-list.md`
- generate `workspace/survey/refs.bib`
- optionally discover open-source repositories for papers

Primary outputs:

- `workspace/paper-list.md`
- `workspace/papers/`
- `workspace/code_repos.md`
- `workspace/survey/refs.bib`

### Reader-1, Reader-2, Reader-3

The three Reader agents parallelize paper analysis.
Their responsibilities are:

- read assigned papers in full
- prefer `paper.md` when LaTeX was converted successfully
- fall back to document-loader MCP for PDFs
- inspect raw `source/*.tex` for equations, figures, and tables
- write structured notes into batch-specific note directories
- store paper metadata and cross-paper relationships in memory MCP

Primary outputs:

- `workspace/notes/batch-1/`
- `workspace/notes/batch-2/`
- `workspace/notes/batch-3/`

### Writer

The Writer synthesizes the results into publication artifacts.
Its responsibilities are:

- read notes across all batches
- query memory MCP for relationships and themes
- build a taxonomy using sequential thinking
- write `workspace/taxonomy.md`
- draft `workspace/survey/main.tex`
- generate `workspace/slides/slides.tex`
- compile the survey and slide deck

Primary outputs:

- `workspace/taxonomy.md`
- `workspace/survey/main.tex`
- `workspace/survey/main.pdf`
- `workspace/slides/slides.tex`
- `workspace/slides/slides.pdf`

## Workflow Stages

The designed workflow is linear with explicit checkpoints:

1. Topic intake
2. Tool and MCP verification
3. Paper discovery and ranking
4. Human approval of paper shortlist
5. Paper download and citation aggregation
6. Parallel reading and note generation
7. Taxonomy synthesis
8. Human approval of taxonomy
9. Survey writing
10. Slide generation
11. Build and final review

This checkpoint structure is intentional.
The system is designed to keep humans in control of scope and scientific framing.

## Skills

The implementation uses four explicit skills.

### `search-papers`

Purpose:

- find candidate papers for a topic
- identify seed papers
- expand via references and citations
- rank and shortlist papers
- download approved papers
- discover code repositories
- merge citations

Key dependencies:

- `fetch` MCP
- `memory` MCP
- `scripts/arxiv-download.sh`
- `scripts/merge-citations.sh`

Key outputs:

- `workspace/paper-list.md`
- `workspace/papers/`
- `workspace/code_repos.md`
- `workspace/survey/refs.bib`

### `read-paper`

Purpose:

- produce structured notes for each paper
- extract equations, figures, tables, implementation details, and limitations
- record paper-to-paper relationships in memory MCP

Key dependencies:

- `document-loader` MCP
- `memory` MCP
- `workspace/papers/*/paper.md`
- `workspace/papers/*/source/*.tex`

Key outputs:

- `workspace/notes/batch-{N}/{arxiv_id}.md`

### `write-survey`

Purpose:

- aggregate notes and memory
- define the field taxonomy
- draft the survey paper in LaTeX
- generate comparison tables and code availability tables

Key dependencies:

- `memory` MCP
- `sequentialthinking` MCP
- `workspace/notes/`
- `workspace/survey/refs.bib`
- `.claude/skills/write-survey/templates/survey.tex`

Key outputs:

- `workspace/taxonomy.md`
- `workspace/survey/main.tex`

### `make-slides`

Purpose:

- transform the completed survey into a presentation deck
- preserve the taxonomy structure in slide form
- compile a Beamer PDF

Key dependencies:

- `workspace/survey/main.tex`
- `workspace/survey/main.pdf`
- `workspace/taxonomy.md`
- `.claude/skills/make-slides/templates/beamer.tex`

Key outputs:

- `workspace/slides/slides.tex`
- `workspace/slides/slides.pdf`

## External Runtime Components

### MCP Servers

The repo configures five MCP servers in `.mcp.json`:

- `document-loader`
- `fetch`
- `git`
- `memory`
- `sequentialthinking`

These servers provide the nontrivial runtime capabilities the workflow depends on:

- PDF extraction with equation preservation
- HTTP access to search APIs
- persistent memory across tasks
- structured reasoning for taxonomy synthesis
- optional version-control integration

### Local Tools

The workflow also assumes the following local command-line tools exist:

- `claude`
- `uvx`
- `npx`
- `pandoc`
- `pdflatex`
- `bibtex`
- `make`

## File Ownership and Data Flow

The design uses directory-based ownership to reduce conflicts between agents.

- Researcher writes to `workspace/papers/`, `workspace/paper-list.md`, and `workspace/survey/refs.bib`
- Readers write only to `workspace/notes/batch-{N}/`
- Writer writes to `workspace/taxonomy.md`, `workspace/survey/`, and `workspace/slides/`

The intended data flow is:

1. discovered metadata -> `paper-list.md`
2. approved downloads -> `workspace/papers/`
3. citations -> `workspace/survey/refs.bib`
4. notes -> `workspace/notes/`
5. taxonomy -> `workspace/taxonomy.md`
6. survey draft -> `workspace/survey/main.tex`
7. slides -> `workspace/slides/slides.tex`
8. final PDFs -> `workspace/survey/main.pdf` and `workspace/slides/slides.pdf`

## Shell Helpers and Build System

### `scripts/arxiv-download.sh`

This script is a convenience helper for:

- validating the arXiv identifier
- fetching title metadata
- creating a per-paper destination directory
- downloading BibTeX
- downloading the PDF
- downloading e-print source
- extracting or copying LaTeX source into `source/`
- attempting a `pandoc` conversion to `paper.md`

### `scripts/merge-citations.sh`

This script merges `citation.bib` files from each downloaded paper into:

- `workspace/survey/refs.bib`

### Makefiles

The survey and slides each have a small Makefile that:

- runs `pdflatex`
- runs `bibtex`
- reruns `pdflatex` to resolve references

These files are intentionally simple and are meant to be easy to inspect or replace.

## Human Checkpoints

Three human approval points are part of the design:

1. shortlist approval after paper discovery
2. taxonomy approval after synthesis
3. final review after compilation

This is one of the central design choices in the repo.
The system is optimized for assisted autonomy rather than full autonomy.

## Templates and Artifacts

The repo ships two templates:

- `.claude/skills/write-survey/templates/survey.tex`
- `.claude/skills/make-slides/templates/beamer.tex`

These act as the default output skeletons for:

- an IEEE-style survey paper
- a Beamer slide deck

They are meant to be edited for different publication formats or presentation styles.

## Operational Assumptions

- The user runs the workflow from an installed project directory
- MCP servers can be launched successfully
- network access to upstream services is available
- papers are primarily sourced from arXiv
- the user is willing to review and approve intermediate checkpoints

## Current Caveats

The current implementation is a usable scaffold, but it is not fully hardened.
Known caveats include:

- `scripts/arxiv-download.sh` currently accepts only modern arXiv IDs in `YYMM.NNNNN` form, so legacy IDs such as `archive/YYMMNNN` are not supported
- the pipeline depends heavily on external services and metadata quality
- build and download helpers are intentionally lightweight and may need hardening for broader production use

## Extension Points

Likely customization points include:

- replacing the LaTeX survey template
- replacing the Beamer template
- changing the ranking or filtering strategy in `search-papers`
- changing the note schema in `read-paper/template.md`
- adding new MCP tools or replacing existing ones
- adding support for non-arXiv sources
- improving downloader robustness and identifier coverage

## Summary

ClaudePaperSurveyAgent is a multi-agent research-writing scaffold for Claude Code.
Its core design is:

- specialized agents with clear file ownership
- skill-based workflows for each stage
- MCP-backed retrieval, memory, and reasoning
- LaTeX-first output generation
- human approval gates to keep the system grounded
