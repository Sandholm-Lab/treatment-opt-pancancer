#!/bin/bash
# a runner script to execute the multi-cell experiment with varying lambda on the simplex.

cmd=""
seed=23
for value in {4..28}
do
cmd+="python experiments/multi_cell.py -t skin -d simplex -o worst -r $seed -l $(echo "$value / -4" |bc -l)\n"
cmd+="python experiments/multi_cell.py -t intestine -d simplex -o worst -r $seed -l $(echo "$value / -4" |bc -l)\n"
cmd+="python experiments/multi_cell.py -t pancreas -d simplex -o worst -r $seed -l $(echo "$value / -4" |bc -l)\n"
cmd+="python experiments/multi_cell.py -t breast -d simplex -o worst -r $seed -l $(echo "$value / -4" |bc -l)\n"
done
echo -e $cmd
echo "============================="
echo "Waiting 10s before starting jobs...."
sleep 10s
echo -e $cmd | xargs -n1 -P7 -I{} -- bash -c '{}'
