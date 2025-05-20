# #!/bin/bash
# # run this script with: nohup bash submit_run_timing.sh > out.out &
# export CUDA_VISIBLE_DEVICES=0
# # Array of durations
# durations=(4.0)

# # Iterate over each duration
# for duration in "${durations[@]}"; do
#     # Construct output filenames
#     output_file="final_timing_${duration}yr"
#     log_file="final_timing_log_${duration}yr"

#     # Run the Python script with the specified duration and additional flags
#     python run_timing.py -v -f "$output_file" --generate_parameters --nsamples 1000 --epsilon --duration "$duration" --iterations 1 -l "$log_file"

#     echo "Completed run for duration ${duration} year(s). Results saved to ${output_file}.json and ${log_file}.log"
# done


# # Execute the Python command and redirect output to the dynamic filename
# # condor_submit -a "duration=$duration" -a "output_file=$output_file" -a "log_file=$log_file" -a "nsamples=1000" -a "epsilon=true" -a "iterations=1" submit_file.submit

#!/bin/bash
# Usage: bash submit_run_timing_jobs.sh
# /data/lsperi/KerrEccentricEquatorialFiguresRestructure/scripts/Results/timing
# export CUDA_VISIBLE_DEVICES=3
durations=(4.0)

for duration in "${durations[@]}"; do
    output_file="lakshmi_timing_${duration}yr"
    log_file="lakshmi_timing_log_${duration}yr"
    condor_submit \
        -a "duration=$duration" \
        -a "output_file=$output_file" \
        -a "log_file=$log_file" \
        submit_run_timing.submit
done