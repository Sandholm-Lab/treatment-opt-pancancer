#!/bin/bash
# a runner script to execute the sequential 2-step experiments for intestine cancers.


cmd=""
seed=23
for value in {4..28}
do
cmd+="python experiments/sequential.py -t intestine -d simplex -s 2 -o worst -r $seed -l $(echo "$value / -4" |bc -l)\n"
cmd+="python experiments/sequential_baseline.py -t intestine -d simplex -s 2 -o worst -r $seed -l $(echo "$value / -4" |bc -l)\n"
done
echo -e $cmd | xargs -0 -I{} -- bash -c '{}' -P 7
