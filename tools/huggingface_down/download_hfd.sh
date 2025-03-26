#!/bin/bash
# Usage:

#    hfd <REPO_ID> [--include include_pattern1 include_pattern2 ...] [--exclude exclude_pattern1 exclude_pattern2 ...] [--hf_username username] 
#    \n [--hf_token token] [--tool aria2c|wget] [-x threads] [-j jobs] [--dataset] [--local-dir path] [--revision rev]

# Ref hfd from: https://hf-mirror.com/

sudo apt install aria2c
export HF_ENDPOINT=https://hf-mirror.com

# Set the parameters
model_id="black-forest-labs/FLUX.1-Fill-dev"
# exclude_pattern="*.safetensors"
hf_username="YOUR HF NAME"
hf_token="YOUR HF TOKEN"
tool="aria2c"
threads="10"
concurrent_file='1'
dir='YOUR_CACHE_PATH/black-forest-labs/FLUX.1-Fill-dev'

bash hfd.sh $model_id \
    --hf_username $hf_username \
    --hf_token $hf_token \
    --tool $tool \
    -x $threads \
    -j $concurrent_file \
    --local-dir $dir \
    # --exclude $exclude_pattern \
    # --dataset
