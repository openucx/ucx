---
name: ucx-pr-review
description: Review UCX pull requests with the project review style, GitHub workflow, and reviewer checklist. Use when asked to review a UCX PR, inspect a GitHub pull request, draft review comments, triage PR scope, or apply REVIEW.md guidance.
---

# UCX PR Review

## Overview

Use this skill to review UCX pull requests. It layers GitHub operating
procedure on top of the repository review checklist in `REVIEW.md`.

## Sources

Read these files before making review claims:

- `AGENTS.md`
- Relevant subtree `AGENTS.md` files for changed paths
- `REVIEW.md`
- `docs/CodeStyle.md`, `docs/LoggingStyle.md`, and
  `docs/OptimizationStyle.md` when the diff touches those concerns

## GitHub Workflow

1. Identify the base branch, changed files, added/deleted line count, CI state,
   and author intent from the PR title and description.
2. Inspect the diff before inspecting unrelated code. Use surrounding code only
   to verify a suspected issue or local convention.
3. Check whether the PR should be split: bug fixes must not include unrelated
   refactoring, and large changes should be called out early.
4. Review by risk: API/ABI, wire compatibility, correctness, error handling,
   cleanup, concurrency, hot paths, tests, build files, docs, then style.
5. Before posting a finding, verify the exact line and confirm the issue is not
   already raised or explained in the thread.

If the `gh` CLI is available and the user permits network access, useful
commands are:

```sh
gh pr view <PR> --json title,body,baseRefName,headRefName,additions,deletions,changedFiles,reviewThreads,comments
gh pr diff <PR>
gh pr checks <PR>
```

## Comment Rules

- Lead with findings. Keep summaries short and secondary.
- Use the tone and severity rules from `REVIEW.md`.
- Prefer concise questions or direct statements. Most comments should be one or
  two sentences.
- Use `blocker:` only for must-fix defects such as crash, corruption, ABI/API
  breakage, broken CI, or clear correctness regressions.
- Use `minor:` for style-only or readability nits.
- Do not bundle unrelated issues into one comment.
- Avoid broad style comments unless they are grounded in changed lines.

## Output

When returning a review in chat, list findings first with file and line
references. If there are no findings, say so and note any residual test or
hardware coverage gaps.
- PDF skill: `fill_fillable_fields.py`, `extract_form_field_info.py` - utilities for PDF manipulation
- DOCX skill: `document.py`, `utilities.py` - Python modules for document processing

**Appropriate for:** Python scripts, shell scripts, or any executable code that performs automation, data processing, or specific operations.

**Note:** Scripts may be executed without loading into context, but can still be read by Codex for patching or environment adjustments.

### references/
Documentation and reference material intended to be loaded into context to inform Codex's process and thinking.

**Examples from other skills:**
- Product management: `communication.md`, `context_building.md` - detailed workflow guides
- BigQuery: API reference documentation and query examples
- Finance: Schema documentation, company policies

**Appropriate for:** In-depth documentation, API references, database schemas, comprehensive guides, or any detailed information that Codex should reference while working.

### assets/
Files not intended to be loaded into context, but rather used within the output Codex produces.

**Examples from other skills:**
- Brand styling: PowerPoint template files (.pptx), logo files
- Frontend builder: HTML/React boilerplate project directories
- Typography: Font files (.ttf, .woff2)

**Appropriate for:** Templates, boilerplate code, document templates, images, icons, fonts, or any files meant to be copied or used in the final output.

---

**Not every skill requires all three types of resources.**
