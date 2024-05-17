#!/bin/bash
#
# Zhenhao Ge, 2024-05-09

ROOT_DIR=/home/users/zge/code/repo/parler-tts
DATA_DIR=$ROOT_DIR/examples/parler-tts-demo

CURRENT_DIR=$PWD
[[ $CURRENT_DIR != $ROOT_DIR ]] && cd $ROOT_DIR \
  && echo "change current dir to: $ROOT_DIR"

manifest_file=$DATA_DIR/manifest.json
model_path=$ROOT_DIR/models/parler_tts_mini_v0.1
output_path=$DATA_DIR/wav
num_copies=2
device=0

mkdir -p $output_path

echo "manifest file: ${manifest_file}"
echo "model path: ${model_path}"
echo "output path: ${output_path}"
echo "num copies: ${num_copies}"
echo "GPU device: ${device}"

python infer.py \
  --manifest-file $manifest_file \
  --model-path $model_path \
  --output-path $output_path \
  --num-copies $num_copies \
  --device $device