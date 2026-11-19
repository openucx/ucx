#!/usr/bin/env python3
#
# Copyright (C) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#
# Update copyright year (or end year in a range) to the current year in source
# files touched by a single commit or by a two-commit range (git diff A..B).
#
# With no commit arguments, the script uses git diff @{u}..HEAD (upstream vs
# current branch). If the working directory is not inside a repository, it
# chdirs to the directory containing this script and tries again.
#
# You must pass --nvidia or --holder REGEX: each contributor updates their own
# holder's lines. --nvidia handles NVIDIA CORPORATION & AFFILIATES headers; --holder
# bumps YYYY-YYYY on Copyright lines matching REGEX (excluding bundled khash etc.).
#
# Deps:
#    $ pip install GitPython
#
from git import Repo  # pip install GitPython
from git.exc import GitCommandError, InvalidGitRepositoryError
from optparse import OptionParser
import io
import os
import re
import sys
from datetime import datetime


class CopyrightYearUpdater(object):
    def parse_args(self, argv):
        parser = OptionParser(
            usage="usage: %prog [options] [<commit> [<commit>]]\n"
            "       %prog [options] [<commit1>..<commit2>]\n"
            "\n"
            "If no commits are given, uses diff @{u}..HEAD (upstream vs HEAD).\n"
            "Requires --nvidia or --holder REGEX (no default; pick your holder)."
        )
        parser.add_option(
            "--dry-run",
            action="store_true",
            dest="dry_run",
            default=False,
            help="Print paths that would change but do not write files.",
        )
        parser.add_option(
            "--year",
            type="int",
            dest="year",
            default=None,
            metavar="YYYY",
            help="Year to set [default: current calendar year]",
        )
        parser.add_option(
            "--nvidia",
            action="store_true",
            dest="nvidia",
            default=False,
            help="Only rewrite NVIDIA CORPORATION & AFFILIATES headers.",
        )
        parser.add_option(
            "--holder",
            type="string",
            dest="holder",
            default=None,
            metavar="REGEX",
            help="Only bump YYYY-YYYY on Copyright lines matching REGEX (e.g. org name).",
        )
        (options, args) = parser.parse_args(argv)

        if options.nvidia and options.holder:
            parser.error("--nvidia and --holder are mutually exclusive")

        holder_pat = (options.holder or "").strip()
        if not options.nvidia and not holder_pat:
            parser.error("must specify --nvidia or --holder REGEX")

        self.holder_re = None
        if holder_pat:
            try:
                self.holder_re = re.compile(holder_pat)
            except re.error as err:
                parser.error("invalid --holder regex: %s" % err)

        self.dry_run = options.dry_run
        self.year = options.year
        self.nvidia = options.nvidia
        self.commits = args if args else None

    def _bump_nvidia_copyright_line(self, line, year):
        """
        Bump years in NVIDIA CORPORATION & AFFILIATES lines (--nvidia).

        Order: Copyright (C) … before (c) … AFFILIATES, … (ranges before singles
        are handled inside each unified pattern).
        """
        ys = str(year)
        orig = line

        def sub_copyright_c(m):
            pfx, y1, y2, sep = m.group(1), m.group(2), m.group(3), m.group(4)
            end = y2 if y2 else y1
            if end == ys:
                return m.group(0)
            return pfx + y1 + "-" + ys + sep + "NVIDIA CORPORATION & AFFILIATES."

        line = re.sub(
            r"(Copyright \(C\) )(\d{4})(?:-(\d{4}))?((?:, | ))NVIDIA CORPORATION & AFFILIATES\.",
            sub_copyright_c,
            line,
        )

        def sub_affiliates_comma(m):
            pfx, y1, y2, dot = m.group(1), m.group(2), m.group(3), m.group(4)
            end = y2 if y2 else y1
            if end == ys:
                return m.group(0)
            return pfx + y1 + "-" + ys + dot

        line = re.sub(
            r"(NVIDIA CORPORATION & AFFILIATES, )(\d{4})(?:-(\d{4}))?(\.)",
            sub_affiliates_comma,
            line,
            flags=re.IGNORECASE,
        )

        return line if line != orig else orig

    def _bump_generic_year_ranges(self, line, year):
        """
        With --holder: on lines containing 'Copyright', replace YYYY-ZZZZ by
        YYYY-<year> when the end year differs.
        """
        if not re.search(r"copyright", line, re.IGNORECASE):
            return line

        ys = str(year)

        def repl(m):
            if m.group(2) != ys:
                return "%s-%s" % (m.group(1), ys)
            return m.group(0)

        return re.sub(r"\b(19\d{2}|20\d{2})-(19\d{2}|20\d{2})\b", repl, line)

    def _process_file(self, path, year):
        try:
            with io.open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except (OSError, UnicodeDecodeError):
            return False

        out_lines = []
        changed = False
        for line in text.splitlines(True):
            if self.nvidia:
                new_line = self._bump_nvidia_copyright_line(line, year)
            else:
                if not self.holder_re.search(line):
                    new_line = line
                else:
                    new_line = self._bump_generic_year_ranges(line, year)
            if new_line != line:
                changed = True
            out_lines.append(new_line)

        if changed and not self.dry_run:
            with io.open(path, "w", encoding="utf-8") as f:
                f.write("".join(out_lines))
        return changed

    def _open_repo(self):
        try:
            return Repo(search_parent_directories=True)
        except InvalidGitRepositoryError:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(script_dir)
            return Repo(search_parent_directories=True)

    def main(self, argv):
        self.parse_args(argv[1:])

        try:
            repo = self._open_repo()
        except InvalidGitRepositoryError:
            sys.stderr.write(
                "Error: not a git repository (or any of the parent directories)\n"
            )
            return 1

        year = self.year if self.year is not None else datetime.now().year
        commits = self.commits

        try:
            if commits is None:
                raw = repo.git.diff("--name-only", "@{u}..HEAD")
            elif len(commits) == 1:
                spec = commits[0]
                if ".." in spec and spec.count("..") == 1 and "..." not in spec:
                    a, b = spec.split("..", 1)
                    if not a:
                        sys.stderr.write(
                            "Error: Invalid range: use A..B with both endpoints.\n"
                        )
                        return 2
                    raw = repo.git.diff("--name-only", "%s..%s" % (a, b))
                else:
                    raw = repo.git.diff_tree(
                        "--no-commit-id", "--name-only", "-r", spec
                    )
            elif len(commits) == 2:
                raw = repo.git.diff(
                    "--name-only", "%s..%s" % (commits[0], commits[1])
                )
            else:
                sys.stderr.write(
                    "Error: Provide one commit (or A..B range), or two commits.\n"
                )
                return 2
        except GitCommandError as e:
            stderr = getattr(e, "stderr", None) or b""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            msg = (stderr or str(e)).strip()
            sys.stderr.write("git failed: %s\n" % msg)
            return 1

        files = [ln for ln in raw.splitlines() if ln.strip()]
        root = repo.working_tree_dir
        updated = 0
        skipped = 0

        for rel in files:
            path = os.path.join(root, rel)
            if not os.path.isfile(path):
                skipped += 1
                continue
            if self._process_file(path, year):
                updated += 1
                print(
                    "%s: %s"
                    % (
                        "would update" if self.dry_run else "updated",
                        rel,
                    )
                )

        print(
            "Done: %d file(s) with copyright changes, %d skipped "
            "(missing or not a regular file). Year=%d."
            % (updated, skipped, year)
        )
        return 0


if __name__ == "__main__":
    rc = CopyrightYearUpdater().main(sys.argv)
    sys.exit(rc)
