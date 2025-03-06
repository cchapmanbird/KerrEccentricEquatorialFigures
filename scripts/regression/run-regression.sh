#! /usr/bin/bash -l

# Usage: Given two environments with FEW, few_rc1 and few_rc2,
# run this script to produce identical data products in each environment and
# then compare them.
#
# ./run-regression.sh few_rc1 few_rc2
#

# relative tolerance
RELTOL=1e-14

# # sometimes this code is needed to load conda into the Bash env
# source deactivate
# source ~/.bashrc

echo "Running regression test for relative tolerance ${RELTOL}"

# # You may want to purge the log files
# echo "deleting old log files"
# rm *.log

conda activate "$1"

echo "generating parameters"

# Generate and save parameters for regression test
python waveform-regression-test.py -g -s 34278 -l 200

conda deactivate

echo "generating waveform data"

for x in $@
do
    echo "running environment ${x}"
    conda activate ${x}
    python waveform-regression-test.py -p testing_parameters.json -o "test_results_${x}"
    conda deactivate
done

conda activate "$1"

python check-results.py -e $RELTOL

