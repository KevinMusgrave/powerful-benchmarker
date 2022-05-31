#!/bin/bash

conda_env=$1
command=$2

echo "Command to be wrapped: $command"

conda deactivate && conda activate ${conda_env}

${command}