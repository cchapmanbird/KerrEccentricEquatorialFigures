#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
# Array of durations
durations=(0.1)

# Iterate over each duration
for duration in "${durations[@]}"; do
    # Construct output filenames
    output_file="test_timing_${duration}yr"
    log_file="test_timing_log_${duration}yr"

    # Run the Python script with the specified duration and additional flags
    python run_timing.py -v -f "$output_file" --generate_parameters --nsamples 10 --epsilon --duration "$duration" --iterations 1 -l "$log_file"

    echo "Completed run for duration ${duration} year(s). Results saved to ${output_file}.json and ${log_file}.log"
done