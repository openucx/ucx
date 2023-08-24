#!/usr/bin/python3

"""
Checks perf test results for regressions based on a specified threshold.
Usage: ./script_name.py <filename> <threshold>
"""

import sys


def print_usage():
    print(f"Usage: {sys.argv[0]} <filename> <threshold>")
    print("\t<filename>: The file containing performance test results.")
    print("\t<threshold>: The percentage threshold to detect significant performance regression.")
    exit(1)


def check_perf_results(filename, threshold):
    max_regression = 0
    test_name_max = test_name = None
    with open(filename, "r") as f:
        for line in f:
            if "+osu" in line:
                test_name = line.split()[0].strip("+")
            elif "% worse" in line:
                worse_value = float(line.split("%")[0].split()[-1])
                if worse_value > max_regression:
                    max_regression = worse_value
                    test_name_max = test_name
    if max_regression > threshold:
        msg = f"Max performance regression detected in {test_name_max}, regression: {max_regression}%"
        print(f"{msg}")
        print(f"##vso[task.complete result=SucceededWithIssues;]DONE${msg}")
    print("No significant performance regression detected.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print_usage()
    check_perf_results(sys.argv[1], float(sys.argv[2]))
