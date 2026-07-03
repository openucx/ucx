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

## Code Checkout

Use a separate shallow clone as reference for surrounding code and related
context from other files in the codebase; do not disturb the user's worktree.

```sh
user_name=$(id -un)
repo_dir=$(mktemp -d "${TMPDIR%/}/ucx-pr-${user_name}-<PR>.XXXXXX")
git clone --depth=1 --branch <base-ref> https://github.com/openucx/ucx.git "$repo_dir"
git -C "$repo_dir" fetch --depth=1 origin pull/<PR>/head:pr-<PR>
git -C "$repo_dir" checkout pr-<PR>
git -C "$repo_dir" rev-parse HEAD
```

## GitHub Workflow

1. Identify the base branch, changed files, added/deleted line count, CI state,
   and author intent from the PR title and description.
2. Read existing PR discussion first and apply the existing-comments rules
   below.
3. Treat the GitHub/app/`gh` PR diff as authoritative for changed files and
   review line anchors.
4. Apply the PR size and scope rules from `REVIEW.md`.
5. Apply the risk order and checklist from `REVIEW.md`.
6. Run the candidate-comment gate below before keeping any finding.
7. Run the review-submission self-check before returning or posting the review.

If the `gh` CLI is available and the user permits network access, useful
commands are:

```sh
gh pr view <PR> --json title,body,baseRefName,headRefName,additions,deletions,changedFiles,comments,latestReviews,files,statusCheckRollup
gh pr diff <PR>
gh pr checks <PR>
```

## Existing Comments

- Avoid adding a comment that duplicates a previous comment.
- If the same issue was already raised in an unresolved thread, add a `+1`
  reaction with `gh api` instead of opening a new thread. For inline PR review
  comments, use:

```sh
gh api -X POST repos/<owner>/<repo>/pulls/comments/<comment-id>/reactions \
  -H "Accept: application/vnd.github+json" \
  -f content="+1"
```

  For top-level PR conversation comments, use
  `repos/<owner>/<repo>/issues/comments/<comment-id>/reactions` instead.
- If the author replied with a good explanation, accept the intent unless the
  current code or CI contradicts it.
- If new evidence changes an existing concern, reply in the existing thread
  instead of opening a duplicate thread.

## Candidate Comment Gate

Keep a finding only after verifying all of these:

- It is not already covered by an existing comment.
- The issue is still present in the latest diff.
- The comment is anchored to a changed line, or clearly belongs as PR-level
  feedback.
- The impact is concrete.
- The severity matches `REVIEW.md`.

## Review Submission

- Submit reviews explicitly with `gh`, not the GitHub app connector.
- Use the tone, severity, and length rules from `REVIEW.md`.
- Accumulate all inline comments and submit them in one review submission. Do
  not submit one GitHub review per comment.
- Leave the review body empty unless there is a real PR-level concern.
- Do not generate boilerplate review summaries, finding counts, severity
  counts, or `Code Review` headings.
- Remove comments about intentional tradeoffs that were already explained,
  comments that are only interesting observations, and any finding whose impact
  is unclear.
- Scrub comments for `REVIEW.md` style violations such as severity headings,
  emoji labels, `[P*]` labels, and uppercase severity labels.
- Check every inline comment for suggestion eligibility. If the requested fix is
  an obvious, small, deterministic replacement of changed lines, use a GitHub
  suggestion block instead of prose-only feedback; add at most one short
  justification sentence after it.
- Downgrade uncertain blockers to questions or no-prefix comments.
- Use `COMMENT` mode when all comments are minor or explicitly non-blocking.
- Use `REQUEST_CHANGES` mode when there are blocker or other must-fix findings.
- Use `APPROVE` mode when there are no comments.
- For review-body-only submissions, use `gh pr review <PR> --repo <owner>/<repo>`
  with exactly one of `--comment`, `--request-changes`, or `--approve`; pass
  non-empty review text with `--body-file <file>` to avoid shell quoting issues.
- For inline comments, create one review through `gh api` and the GitHub
  `POST /repos/{owner}/{repo}/pulls/{pull_number}/reviews` endpoint. Put the
  event (`COMMENT`, `REQUEST_CHANGES`, or `APPROVE`), optional body, and all
  inline comments in a temporary JSON file and submit it with:

```sh
gh api -X POST repos/<owner>/<repo>/pulls/<PR>/reviews --input <review-json>
```

## Output

When returning a review in chat, use one of these formats:

- Findings: list each issue with severity, `file:line`, problem, and impact.
- Draft GitHub comments: provide the exact proposed comment text for each
  changed line.
- No findings: say so and note residual test or hardware coverage gaps.
