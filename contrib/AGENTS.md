# Agent Guide: contrib

Developer scripts and build-helper artifacts. Not part of the UCX runtime.
The most important entries are the `configure-*` shell helpers — agents
should call these instead of inventing their own `./configure` flag sets.

## Configure Helpers

All wrap `./configure` from the source root. Pass any extra flags through
`"$@"`.

- `configure-devel` — full debug build (logging, assertions,
  `--enable-gtest`, `--enable-examples`, `--enable-test-apps`,
  `--enable-stats`, `--enable-debug-data`, `--enable-mt`,
  `--with-valgrind=guess`). **Default for any agent doing dev work — see
  the `ucx-build` skill.**
- `configure-opt` — release-style build with optimizations and reduced
  logging. For perf experiments.
- `configure-prof` — profiling build with frame pointers and the UCS
  profiling subsystem enabled.
- `configure-release` / `configure-release-mt` — packaging configurations
  used by the RPM/Debian build paths. Agents should not modify these
  casually; downstream packaging consumes them.

## Other Subdirectories

- `ucx_perftest_config/` — canned `ucx_perftest` parameter files
  (covered by `README`).
- `mtt/` — Open MPI MTT integration scripts (test harness driver).
- `ibmock/` — userspace IB Verbs mock used to run IB tests without
  hardware (`README.md` here). CI heavily depends on this.
- `cray-ugni-mock/` — analogous mock for Cray uGNI builds.
- `wireshark/` — UCX protocol dissectors.

## Top-level Scripts

- `buildrpm.sh`, `rpmdef.sh.in`, `ucx.in` — RPM build entry points.
- `squash_commit.sh`, `pr_merge_check.py`, `authors_update.sh`,
  `api_update.sh` — git/PR utilities.
- `check_inst_headers.sh`, `check_qps.sh`, `test_namespace.sh`,
  `test_jenkins.sh`, `test_efa.sh` — test/CI helpers, mostly invoked from
  `buildlib/azure-pipelines*.yml`.
- `upload_docs.sh` — publishes generated docs (used by the docs CI job).
- `valgrind.supp`, `lsan.supp` — suppression files referenced from
  `test/gtest/Makefile.am`.
- `ctags.sh`, `ctags_ucx.awk`, `ucx-style.el`, `ucx.vim`,
  `gnu-indent-options` — editor / indexer integration matching
  `docs/CodeStyle.md`.

## Conventions

- Don't invent new `./configure` flag combinations for one-off use —
  add a new `configure-*` helper if it'll recur, otherwise pass extra
  flags through the existing helper's `"$@"`.
- Suppression files are version-controlled because CI consumes them;
  add new suppressions deliberately, with a comment about the upstream
  bug they cover.
- `ibmock/` is a real C codebase (with its own `README.md`); follow
  `docs/CodeStyle.md` when modifying it. `cray-ugni-mock/` is only
  header stubs (`gni_pub.h`, `pmi.h`, `pmi2.h`) and `.pc` files for
  building UCX without uGNI hardware/headers — there are no compiled
  sources there.

## Pointers

- Build skill: `.agents/skills/ucx-build/SKILL.md` (already references
  these helpers).
- CI usage: `buildlib/azure-pipelines*.yml`.
- Style files generated/used here line up with `docs/CodeStyle.md`.
- Note: the fuzzy-match test referenced by some CI jobs lives at
  `test/apps/test_fuzzy_match.py`, not in this directory.
