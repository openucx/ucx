---
name: ucx-codegraph
description: Inspect UCX source with CodeGraph when a local index is available. Use before inspecting, diagnosing, reviewing, or modifying UCX source code.
---

# UCX CodeGraph

Use CodeGraph on a best-effort basis:

1. From the repository root, check whether `.codegraph` exists.
2. If it exists and `codegraph_explore` is available, use the tool before text
   search or direct source reads. Inspect relevant definitions, call paths,
   dependents, and blast radius.
3. If CodeGraph reports that its index is out of date, run `codegraph sync`
   from the repository root and retry once.
4. If `.codegraph` is absent, CodeGraph is unavailable, or the retry fails,
   fall back to `rg` and direct file reads without blocking the task.
