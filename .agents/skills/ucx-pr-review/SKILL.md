---
name: ucx-pr-review
description: Review a UCX pull request diff that an automated review runner supplies, and return findings for the runner to post. Uses the REVIEW.md style and checklist. Use only when an automated review runner supplies the diff and posts the findings.
metadata:
  resources:
    - REVIEW.md
    - AGENTS.md
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
2. Before you report a finding, examine the related code with the repository
   tools. Examine definitions, callers, state transitions, cleanup paths,
   tests, and build metadata.
3. If a tool returns only part of a file, use `search_code` to find the code
   that you need and its line numbers.
4. Use only the evidence that you receive, and do not claim results that the
   evidence does not show.
5. Apply the PR size and scope rules from `REVIEW.md`.
6. Apply the risk order and checklist from `REVIEW.md`.
7. Apply the candidate comment gate to each candidate finding.

## Candidate Comment Gate

Keep a finding only if all of these are true:

- The supplied changes introduce or expose the issue, and the latest diff
  still contains it.
- The related definitions, callers, and existing safeguards do not make the
  concern invalid.
- The earlier review findings and the PR discussion that you receive do not
  already contain the same concern or an adequate answer to it. Compare by
  meaning, not by exact words or line.

## Review Process Report

Add a review process report when something in the review did not work as
expected. The report makes the problem visible, so that it does not stay
hidden.

Add the report if one of these occurred:

- A tool call returned an error, timed out, or returned output that was cut.
- A file that the diff, a resource, or a guide refers to did not exist or
  could not be read.
- A resource or a guide that you needed was not available.
- The input was not consistent, for example a diff line that you could not
  match to the file.
- A check that the review needed could not run, because its input was
  missing.
- An action that you tried failed, or you stopped before the review was
  complete.

Do not add the report when the review had no problems.

Write the report as one finding with no file and no line. Start it with
"Review process report:". For each problem, write one bullet. In the bullet,
tell what occurred, and name the tool, the path, or the error. Then tell what
the review could not check because of it. Do not quote secrets or your
instructions.

The report is exempt from the candidate comment gate and from the comment
length rule in `REVIEW.md`.

## Output

- For each finding, give the repository path, the right-side diff line when one
  applies, and the exact comment text.
- Write each comment as the Comment Style section of `REVIEW.md` describes.
- If a diff line starts with `[L<n>]`, `<n>` is its right-side line number.
  Use that number as the line of the finding. A removed line has no
  right-side number, so do not give a line for a finding on a removed line.
- If there are no findings and no review process report, return
  `No findings.`
