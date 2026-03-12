---
name: make-slides
description: Generate Beamer presentation slides from a completed survey paper. Use after the survey LaTeX is written.
allowed-tools: Bash, Read, Write, Glob, Grep
---

# Slide Generation Methodology

## Prerequisites

- Completed survey at `workspace/survey/main.tex`
- Compiled survey PDF at `workspace/survey/main.pdf`
- Taxonomy at `workspace/taxonomy.md`

## Step 1: Extract Key Content

Read the survey `main.tex` and extract:
- Title, authors, date
- Section structure
- Key figures and tables
- Most important equations (1-2 per section)
- Comparison tables
- Open problems

## Step 2: Design Slide Structure

A good survey presentation follows this structure:

```
1. Title slide
2. Motivation / Why this survey? (1-2 slides)
3. Scope and methodology (1 slide)
4. Background / Preliminaries (1-2 slides)
5. Taxonomy overview (1 slide — the "map")
6-15. Main content (2-3 slides per taxonomy category)
   - Each category: overview → key methods → comparison table → key findings
16. Discussion / Cross-cutting themes (1-2 slides)
17. Open problems and future directions (1-2 slides)
18. Conclusion / Key takeaways (1 slide)
19. References (1 slide — selected key references only)
```

Target: 15-20 slides total.

## Step 3: Write Beamer LaTeX

Copy the template:
```bash
cp $CLAUDE_SKILL_DIR/templates/beamer.tex workspace/slides/slides.tex
cp workspace/survey/refs.bib workspace/slides/refs.bib
```

### Slide Writing Rules
- Maximum 5-6 bullet points per slide
- Include key equations (1-2 per technical slide)
- Use `\cite{key}` for attributions — same keys as the survey
- Include comparison tables (simplified for slides)
- Use `\begin{columns}` for side-by-side layouts
- Keep text concise — slides support the talk, not replace the paper

## Step 4: Compile

```bash
make -C workspace/slides
```

Check for compilation errors. Fix any issues.
