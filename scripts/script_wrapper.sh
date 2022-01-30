#!/bin/bash

experiment_name=$1
timeout=$2
root_folder=$3
experiment_folder="$root_folder/$experiment_name/"
conda_env=$4
devices=$5
command=$6

echo "Command to be wrapped: $command"

export CUDA_VISIBLE_DEVICES=${devices}

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

conda deactivate && conda activate ${conda_env}

while [ ! -f "$experiment_folder/best_trial.json" ]
do
	echo "STARTING $script_name"
	echo "Checking folder: $experiment_folder"
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
			if [ -d "$experiment_folder" ]; then
				is_running=$(bash ./scripts/process_checker.sh ${experiment_folder} ${timeout})
			fi
	done
	pkill -9 -P ${curr_pid}
	sleep 10s
done