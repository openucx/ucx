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

- `REVIEW.md`; it points to the repository guides and style docs to load for
  changed paths.

## GitHub Workflow

1. Identify the base branch, changed files, added/deleted line count, CI state,
   and author intent from the PR title and description.
2. Read existing PR discussion first and apply the `REVIEW.md` guidance on
   duplicate, answered, or resolved issues. On GitHub, if the same issue is
   already raised in an unresolved thread, add `:+1:` to the existing comment
   instead of opening a new thread.
3. Inspect the diff before inspecting unrelated code. Use surrounding code only
   to verify a suspected issue or local convention.
4. Apply the PR size and scope rules from `REVIEW.md`.
5. Apply the risk order and checklist from `REVIEW.md`.
6. Before posting a finding, verify the exact changed line and current behavior.

If the `gh` CLI is available and the user permits network access, useful
commands are:

```sh
gh pr view <PR> --json title,body,baseRefName,headRefName,additions,deletions,changedFiles,comments,latestReviews,files,statusCheckRollup
gh pr diff <PR>
gh pr checks <PR>
```

## Comment Rules

- Use the tone, severity, and length rules from `REVIEW.md`.

## Output

When returning a review in chat, use one of these formats:

- Findings: list each issue with severity, `file:line`, problem, and impact.
- Draft GitHub comments: provide the exact proposed comment text for each
  changed line.
- No findings: say so and note residual test or hardware coverage gaps.

For each finding, say whether to post a new comment or add `:+1:` to an
existing unresolved thread.
