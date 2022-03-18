#!/bin/bash

input_folder=$1
timeout=$2
latest_dir=$(ls -td ${input_folder}/*/ | head -1)
latest_update_time=$(find ${latest_dir} -printf "%T@\n" | sort | tail -1 | cut -f1 -d".")
current_seconds=$(date +%s)
time_since_latest_update=$((current_seconds - latest_update_time))
# minutes_since_latest_update=$((time_since_latest_update / 60))
if (( time_since_latest_update > ${timeout} ));
then 
    echo 0;
else
    echo 1;
fi;