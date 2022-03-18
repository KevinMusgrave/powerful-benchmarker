#!/bin/bash

conda_env=$1
devices=$2
command=$3

echo "Command to be wrapped: $command"

export CUDA_VISIBLE_DEVICES=${devices}

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

conda deactivate && conda activate ${conda_env}

${command}