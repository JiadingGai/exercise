---
name: write-survey
description: Write a LaTeX survey paper from structured paper notes and taxonomy. Use when synthesizing a survey from research notes.
allowed-tools: Bash, Read, Write, Glob, Grep, mcp__memory__search_nodes, mcp__memory__open_nodes, mcp__sequentialthinking__sequentialthinking
---

# Survey Writing Methodology

## Prerequisites

Before writing, ensure you have:
- All paper notes in `workspace/notes/*/`
- `workspace/survey/refs.bib` (merged citations)
- The survey template at `$CLAUDE_SKILL_DIR/templates/survey.tex`

## Step 1: Gather All Knowledge

1. Read all note files: `workspace/notes/**/*.md`
2. Query memory MCP for paper relationships:
   ```
   search_nodes("paper")
   ```
3. Read `workspace/survey/refs.bib` — note all available citation keys

## Step 2: Build Taxonomy

Use sequentialthinking MCP to reason through the taxonomy:

Think through:
- What are the major themes/categories across all papers?
- How do papers cluster by method, application, or contribution type?
- What is the chronological evolution of the field?
- What are the open problems and future directions?

Write the taxonomy to `workspace/taxonomy.md`:
```markdown
# Survey Taxonomy: {topic}

## Category 1: {name}
- {paper_key_1}: {one-line relevance}
- {paper_key_2}: {one-line relevance}

## Category 2: {name}
...

## Cross-cutting Themes
- {theme 1}: papers [{keys}]
- {theme 2}: papers [{keys}]

## Open Problems
1. {open problem}
2. {open problem}
```

**PAUSE for human approval of taxonomy before proceeding.**

## Step 3: Write LaTeX

Copy the survey template:
```bash
cp $CLAUDE_SKILL_DIR/templates/survey.tex workspace/survey/main.tex
```

Then fill in each section. Write one section at a time.

### Writing Rules
- One sentence per line (clean git diffs)
- Only use `\cite{key}` for keys that exist in `refs.bib`
- Never fabricate statistics or claims — use placeholders `XX.X\%` if uncertain
- Include LaTeX equations from paper notes (they're already in LaTeX notation)
- Reference figures and tables with `\label{}` and `\ref{}`
- Mark uncertain claims with `% TODO: verify`

### Section Writing Order
1. Abstract (write last, after all sections)
2. Introduction — motivation, scope, contributions of the survey
3. Background — foundational concepts needed to understand the survey
4. Main body sections — one per taxonomy category
5. Discussion — cross-cutting themes, comparison tables
6. Open problems and future directions
7. Conclusion
8. Abstract (now write it)

## Step 4: Create Comparison Tables

### Method Comparison
For each taxonomy category, create a comparison table:

### Code Availability Table
Include a table summarizing open-source implementations:
```latex
\begin{table}[t]
\centering
\caption{Open-source implementations of surveyed methods}
\label{tab:code-availability}
\begin{tabular}{llll}
\toprule
Paper & Repository & Framework & Stars \\
\midrule
\cite{paper1} & \url{github.com/...} & PyTorch/CUDA & 15K \\
\cite{paper2} & \url{github.com/...} & Triton & 2K \\
\cite{paper3} & --- & --- & --- \\
\bottomrule
\end{tabular}
\end{table}
```

Read `workspace/code_repos.md` and paper notes for code information.

### Per-Category Comparison
For each taxonomy category, create a comparison table:
```latex
\begin{table}[t]
\centering
\caption{Comparison of methods in {category}}
\label{tab:comparison-{category}}
\begin{tabular}{lccc}
\toprule
Method & Metric 1 & Metric 2 & Year \\
\midrule
\cite{paper1} & value & value & 2023 \\
\cite{paper2} & value & value & 2024 \\
\bottomrule
\end{tabular}
\end{table}
```

## Step 5: Compile

```bash
make -C workspace/survey
```

Check for compilation errors. Fix any undefined citation warnings.
