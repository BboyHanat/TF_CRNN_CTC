#!/usr/bin/env bash
relative_path=$(pwd)
echo $relative_path
export CUDA_VISIBLE_DEVICES=0
source activate cvpy36
export PYTHONPATH=$relative_path

