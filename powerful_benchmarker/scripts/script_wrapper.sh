#!/bin/bash

exp_name=$1
timeout=$2
exp_folder=$3
full_path="$exp_folder/$exp_name/"
conda_env=$4
devices=$5
best_trial_filename=$6
command=$7

echo "Command to be wrapped: $command"

export CUDA_VISIBLE_DEVICES=${devices}

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

conda deactivate && conda activate ${conda_env}

best_trial_full_path="$full_path$best_trial_filename"

echo "Will run until this file is made: $best_trial_full_path"

while [ ! -f $best_trial_full_path ]
do
	echo "STARTING $script_name"
	echo "Checking folder: $full_path"
	echo "Will kill script if folder has not updated in the past $timeout seconds"
	${command} & 
	curr_pid=$!
	is_running=1
	echo "curr_pid is $curr_pid"
	trap "kill -9 $curr_pid & exit" INT
	sleep 5m

	while ((is_running == 1))
	do
			sleep 1m
			if [ -d "$full_path" ]; then
				is_running=$(bash ./powerful_benchmarker/scripts/process_checker.sh ${full_path} ${timeout})
			fi
	done
	pkill -9 -P ${curr_pid}
	sleep 10s
done