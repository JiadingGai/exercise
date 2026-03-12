---
name: search-papers
description: Search for academic papers on a given topic using arXiv and Semantic Scholar APIs. Use when discovering papers for a survey.
allowed-tools: Bash, Read, Write, Glob, Grep, mcp__fetch__fetch, mcp__memory__create_entities, mcp__memory__create_relations
---

# Paper Discovery Methodology

## Step 1: Keyword Search

Query both arXiv and Semantic Scholar in parallel:

### arXiv API
```bash
curl -s "http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=50&sortBy=relevance" > workspace/search-arxiv.xml
```

Parse the XML for: title, authors, abstract, arxiv ID, published date.

### Semantic Scholar API
Use the fetch MCP tool:
```
fetch("https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=50&fields=title,authors,abstract,citationCount,year,externalIds,tldr")
```

## Step 2: Auto-Discover Seed Papers

From combined results:
1. Sort by citation count (Semantic Scholar provides this)
2. Read abstracts of top 10 most-cited papers
3. Identify 2-3 foundational/seminal papers as seeds
4. Store seeds in memory MCP:
   ```
   create_entities([{name: "Paper Title", entityType: "seed_paper", observations: ["arxiv:XXXX.XXXXX", "citations: N", "key contribution: ..."]}])
   ```

## Step 3: Citation Graph Expansion

For each seed paper, fetch papers that cite it:
```
fetch("https://api.semanticscholar.org/graph/v1/paper/ArXiv:{arxiv_id}/citations?fields=title,abstract,citationCount,year,externalIds&limit=100")
```

Filter citing papers: keep those whose abstract mentions the survey topic keywords.

Also fetch references of seed papers:
```
fetch("https://api.semanticscholar.org/graph/v1/paper/ArXiv:{arxiv_id}/references?fields=title,abstract,citationCount,year,externalIds&limit=100")
```

## Step 4: Auto-Filter and Rank

For all candidates (~100-200 papers):
1. Read each abstract
2. Score relevance (0-10): Does this paper directly address the survey topic?
3. Compute rank score: relevance × log(1 + citation_count)
4. Sort by rank score
5. Select top 25-30 candidates

## Step 5: Write Paper List

Write `workspace/paper-list.md` with this format:

```markdown
# Paper Candidates for Survey: {topic}

## Recommended (Top 25)

| # | arXiv ID | Title | Authors | Year | Citations | Relevance | Score |
|---|----------|-------|---------|------|-----------|-----------|-------|
| 1 | 2205.14135 | FlashAttention: ... | Tri Dao et al. | 2022 | 5000 | 10 | 37.0 |
| ... |

## Seed Papers (used for citation expansion)
- [seed 1 details]
- [seed 2 details]

## Excluded (with reasons)
- [paper]: excluded because [reason]
```

Present this to the team lead for human approval before proceeding.

## Step 6: Download Approved Papers

For each approved paper:
```bash
bash scripts/arxiv-download.sh {arxiv_id} workspace/papers
```

## Step 7: Convert to Readable Format

For each downloaded paper:
```bash
# Find main .tex file
MAIN_TEX=$(find workspace/papers/{id}_*/source -name "*.tex" -exec grep -l '\\begin{document}' {} \; 2>/dev/null | head -1)

# Convert LaTeX → markdown (preferred)
if [ -n "$MAIN_TEX" ]; then
  cd "$(dirname "$MAIN_TEX")"
  pandoc "$(basename "$MAIN_TEX")" -t markdown -o "../../paper.md" 2>/dev/null
  cd -
fi
```

For papers without LaTeX source, use document-loader MCP to read the PDF.

## Step 8: Discover Code Repositories

For each paper, find its open-source code:

### Primary: Papers With Code API
```
fetch("https://paperswithcode.com/api/v1/papers/?arxiv_id={arxiv_id}")
```
Returns: `{results: [{paper: {url_abs, ...}, repositories: [{url, stars, framework}]}]}`

### Secondary: Grep LaTeX source and converted markdown
```bash
grep -rioP 'https?://github\.com/[^\s\}\)\"]+' workspace/papers/{id}_*/source/ workspace/papers/{id}_*/paper.md 2>/dev/null | head -5
```

### Tertiary: Semantic Scholar
```
fetch("https://api.semanticscholar.org/graph/v1/paper/ArXiv:{arxiv_id}?fields=externalIds,openAccessPdf")
```

For each paper, record the code repository (if found) in memory MCP:
```
create_entities([{name: "{paper_title}", entityType: "paper", observations: ["repo: {github_url}", "stars: {count}", "framework: {pytorch/jax/triton}"]}])
```

Write a `code_repos.md` summary:
```markdown
| Paper | Repository | Stars | Framework |
|-------|-----------|-------|-----------|
| FlashAttention | github.com/Dao-AILab/flash-attention | 15000 | PyTorch/CUDA |
| ... | ... | ... | ... |
| Paper X | ❌ No public code found | — | — |
```

## Step 9: Merge Citations

```bash
bash scripts/merge-citations.sh
```

Store all paper metadata in memory MCP for teammates to query.
