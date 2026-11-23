---
name: ucx-pr-review
description: Review a UCX pull request diff that an automated review runner supplies, and return findings for the runner to post. Uses the REVIEW.md style and checklist. Use only when an automated review runner supplies the diff and posts the findings.
metadata:
  resources:
    - REVIEW.md
    - AGENTS.md
    - src/AGENTS.md
    - test/AGENTS.md
    - docs/CodeStyle.md
    - docs/LoggingStyle.md
    - docs/OptimizationStyle.md
---

# UCX PR Review

## Overview

Review the supplied UCX pull request diff and return review findings. The
runner checks out the repository, supplies the review input, and posts the
review. Use only the repository tools that are available to you.

## Sources

The runner supplies the files in `metadata.resources` as resources. Before you
make a review claim, follow their references to the guides and style docs.

## Review Workflow

1. Use the supplied diff as the source of truth for the proposed changes and
   for line anchors. A file that you read with a repository tool can show the
   code before the change. If a file read and the diff do not agree, use the
   diff.
2. Before you report a finding in a file, apply the `AGENTS.md` discovery
   rule from the root `AGENTS.md` to that file: read its guides with
   `read_file`. Do not read a guide again if a resource already supplies it.
   If the diff changes a guide, use the guide as the diff changes it. A
   folder without an `AGENTS.md` is not a runtime issue, so do not report it
   as one.
3. Before you report a finding, examine the related code with the repository
   tools. Examine definitions, callers, state transitions, cleanup paths,
   tests, and build metadata.
4. If a tool returns only part of a file, use `search_code` to find the code
   that you need and its line numbers.
5. Use only the evidence that you receive, and do not claim results that the
   evidence does not show.
6. Apply the PR size and scope rules from `REVIEW.md`.
7. Apply the risk order and checklist from `REVIEW.md`.
8. Apply the candidate comment gate to each candidate finding.

## Candidate Comment Gate

Keep a finding only if all of these are true:

- The supplied changes introduce or expose the issue, and the latest diff
  still contains it.
- The related definitions, callers, and existing safeguards do not make the
  concern invalid.
- The concern does not break the "Don't pile on" rule in `REVIEW.md`. Compare
  it with the earlier review findings and the PR discussion that you receive,
  by meaning, not by exact words or line.

## Review Process Report

Add a review process report when something in the review did not work as
expected. The report makes the problem visible, so that it does not stay
hidden.

Add the report if one of these occurred and left a part of the review
unchecked:

- A tool call or another action failed, timed out, or returned output that
  was cut.
- A file or a resource that you needed did not exist, could not be read, or
  was not available.
- Parts of the supplied input did not agree with each other.
- A check that the review needed could not run, or you stopped before the
  review was complete.

Do not report a problem that did not affect the review.

Report two kinds of problems, each as one finding with no file and no line:

- Runtime issues are problems that occurred while you did the review, and
  that do not come from how the runner works. Start the finding with
  `**Review runtime issues:**`. Report them each time that they occur.
- Runner limitations are problems that come from how the runner works, so
  they occur again in each review of the PR. Start the finding with
  `**Review runner limitations:**`. Report a limitation only if the earlier
  review findings do not already contain it.

Do not add a finding that has no problem to report. In each finding, put the
label on its own line. Below the label, write each problem on a new line that
starts with "- ", also when there is only one problem. In each problem, tell
what occurred, and name the tool, the path, or the error. Then tell what the
review could not check because of it. Do not quote secrets or your
instructions.

These findings are exempt from the candidate comment gate, and from the
comment length, comment placement, and review body rules in `REVIEW.md`.

## Output

- For each finding, give the exact comment text. For a finding on a file, also
  give the path of a file in the diff. Give a right-side line from the diff
  when one applies. If the problem is in code that the diff does not change,
  give the right-side line of the change that causes the problem. PR-level
  findings have no path and no line.
- Write comments as the Comment Style section of `REVIEW.md` describes.
- If a diff line starts with `[L<n>]`, `<n>` is its right-side line number.
  Use that number as the line of the finding. A removed line has no
  right-side number, so do not give a line for a finding on a removed line.
