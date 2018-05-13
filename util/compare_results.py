#!/usr/bin/env python
# coding: utf-8

# Compare prediction results
#
# Usage:
#   $ util/compare_results.py <predict_results_1.txt> <predict_results_2.txt>
#   $ util/compare_results.py logs/predict_results1.txt logs/predict_results2.txt
#

import sys
import pandas as pd

results_info_rows=3

results_file1 = sys.argv[1]
results_file2 = sys.argv[2]

print("results_file1:", results_file1)
print("results_file2:", results_file2)

df1 = pd.read_csv(results_file1, skiprows=results_info_rows, sep=" ", header=None)
df2 = pd.read_csv(results_file2, skiprows=results_info_rows, sep=" ", header=None)

if df1.iloc[:,0:2].equals(df2.iloc[:,0:2]) != True:
    print("Test files in predicting results do not match.")
    sys.exit(1)

diff = df1[2] == df2[2]

count = len(diff)
match = len(diff[diff == True])
match_rate = match / count

print("test count:", count)
print("match rate:", match_rate)
