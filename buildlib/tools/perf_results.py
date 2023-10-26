#!/usr/bin/python3

"""
Checks perf test results for regressions.
Usage: ./script_name.py <pattern> <threshold>
"""

import glob
import os
import sys


def print_usage():
    print(f"Usage: {sys.argv[0]} <pattern> <threshold>")
    print("\t<pattern>: Glob pattern to match performance test files.")
    print("\t<threshold>: Threshold to detect significant regression.")
    exit(1)


def check_perf_results(pattern, threshold):
    max_regression = 0
    test_name_max = file_name_max = None

    for filename in glob.glob(pattern):
        with open(filename, "r") as f:
            test_name = None
            for line in f:
                if "+osu" in line:
                    test_name = line.split()[0].strip("+")
                elif "% worse" in line:
                    worse_val = float(line.split("%")[0].split()[-1])
                    if worse_val > max_regression:
                        max_regression = worse_val
                        test_name_max = test_name
                        file_name_max = os.path.basename(filename)

    if max_regression > threshold:
        msg = f"Max regression in {test_name_max} from {file_name_max}: {max_regression}%"
        print(msg)
        print(f"##vso[task.complete result=SucceededWithIssues;]DONE{msg}")
    else:
        print("No significant performance regression detected.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print_usage()
    check_perf_results(sys.argv[1], float(sys.argv[2]))
