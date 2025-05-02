#!/bin/bash
# run this script with: nohup bash submit_run_timing.sh > out.out &
export CUDA_VISIBLE_DEVICES=2
# Array of durations
durations=(2.0 4.0)

# Iterate over each duration
for duration in "${durations[@]}"; do
    # Construct output filenames
    output_file="new_timing_${duration}yr"
    log_file="new_timing_log_${duration}yr"

    # Run the Python script with the specified duration and additional flags
    python run_timing.py -v -f "$output_file" --generate_parameters --nsamples 1000 --epsilon --duration "$duration" --iterations 1 -l "$log_file"

    echo "Completed run for duration ${duration} year(s). Results saved to ${output_file}.json and ${log_file}.log"
done