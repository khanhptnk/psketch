import os
import sys


file_name = sys.argv[1]

results = []

with open(file_name) as f:
    for line in f:
        line = line.rstrip()
        if 'Train iter' in line:
            num_interactions = int(line.split(',')[-1].split(' = ')[-1].split(' / ')[0])
            i_iter = int(line.split(':')[3].split()[2])
        if 'Evaluation on' in line:
            success_rate = float(line.split(':')[-1].split(',')[0].split(' = ')[-1])
            results.append((i_iter, num_interactions, success_rate))

N = 10000
for i in range(N):
    for x in results[i]:
        print(x, end='\t')
    print()
