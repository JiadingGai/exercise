---
name: read-paper
description: Read a research paper and extract structured notes. Use when analyzing papers for a survey.
allowed-tools: Bash, Read, Write, Glob, Grep, mcp__document-loader__read_document, mcp__memory__create_entities, mcp__memory__create_relations, mcp__memory__search_nodes
---

# Paper Reading Methodology

## Startup Assertion

Before reading any papers, verify document-loader MCP is available:
- Call `mcp__document-loader__read_document` on any PDF in your batch
- If it fails or is not available, STOP and message the team lead: "ERROR: document-loader MCP unavailable"
- Do NOT proceed without PDF reading capability. Do NOT use pdftotext (it garbles equations).

## Reading Priority

For each paper in your assigned batch:

1. **paper.md** (pandoc from LaTeX) — best quality, equations preserved as `$...$`
2. **document-loader MCP** on the PDF — use `mcp__document-loader__read_document(path)` for structured extraction with equation preservation
3. **source/*.tex** files — read raw LaTeX for equations, algorithms, figures

## Reading Process

### Pass 1: Overview (from paper.md or PDF)
- Title, authors, year, venue
- Abstract — what's the main contribution?
- Introduction — what problem does it solve?
- Conclusion — what are the key takeaways?

### Pass 2: Technical Details (from source/*.tex)
- Key equations — copy the LaTeX math notation verbatim
- Algorithms — extract pseudocode or algorithm descriptions
- Figures — note figure numbers, captions, and file paths
- Tables — extract key results tables with numbers

### Pass 3: Code & Implementation
- Check `workspace/code_repos.md` for this paper's repository (Researcher may have found it)
- If not found, search within the paper text for GitHub/GitLab URLs:
  ```bash
  grep -i "github.com\|gitlab.com\|code.*available\|implementation.*available" workspace/papers/{id}_*/paper.md workspace/papers/{id}_*/source/*.tex 2>/dev/null
  ```
- If a repo exists, note: URL, language, framework, stars, key implementation details
- If no repo found, mark as "❌ No public code found"

### Pass 4: Analysis
- How does this paper relate to other papers in the survey?
- Does it extend, contradict, or complement other work?
- What are its limitations?
- What is its specific relevance to the survey topic?

## Output: Structured Notes

For each paper, write a note file using the template at `$CLAUDE_SKILL_DIR/template.md`.

Save to: `workspace/notes/batch-{N}/{arxiv_id}.md`

## Memory Storage

After reading each paper, store key findings in memory MCP:

```
create_entities([{
  name: "{paper_title}",
  entityType: "paper",
  observations: [
    "arxiv:{arxiv_id}",
    "year:{year}",
    "cite_key:{bibtex_key}",
    "contribution: {one-line summary}",
    "method: {key method}",
    "result: {key quantitative result}"
  ]
}])
```

Store relationships between papers:

```
create_relations([{
  from: "{paper_A}",
  to: "{paper_B}",
  relationType: "extends|contradicts|uses|compares_with|improves_upon"
}])
```

## Quality Checks

Before marking a paper as read:
- [ ] All template fields are filled (no empty sections)
- [ ] Key equations are in LaTeX notation
- [ ] At least one quantitative result is noted
- [ ] Relevance to survey topic is explicitly stated
- [ ] Relationships to other papers (if known) are stored in memory
