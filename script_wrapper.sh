#!/bin/bash

script_name=$1
experiment_name=$2
experiment_folder="/home/tkm45/NEW_STUFF/experiments/$experiment_name/"

while :
do

	echo "STARTING $script_name"
	bash ${script_name} & 
	curr_pid=$(echo $!)

	sleep 10m
	is_running=1

	while ((is_running == 1))
	do
			sleep 1m
			is_running=$(bash process_checker.sh ${experiment_folder})
	done
	pkill -9 -P ${curr_pid}
	sleep 30s
done