#!/bin/bash

RESULTDIR="./results/viscot/"
DATADIR="./viscot_benchmark/"
script_paths=(
"docvqa"
"flickr30k"
"gqa"
"infographicsvqa"
"openimages"
"textcap"
"textvqa"
"vsr"
"cub"
"dude"
"sroie"
)

CKPT=$1

echo START CoT Detection...
for ((i=0; i<${#script_paths[@]}; i++)); do
    QUESTION_FILE=${script_paths[i]}

    mkdir -p $RESULTDIR/detection/$QUESTION_FILE
    python -m llava.eval.model_cot_det_loader \
        --model-path ./checkpoints/$CKPT \
        --question-file $DATADIR/benchmark_det/$QUESTION_FILE.jsonl \
        --image-folder ./playground/data/ \
        --answers-file $RESULTDIR/detection/$QUESTION_FILE/$CKPT.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1 
done



echo START CoT Answer...

for ((i=0; i<${#script_paths[@]}; i++)); do
    QUESTION_FILE=${script_paths[i]}

    mkdir -p $RESULTDIR/results/$QUESTION_FILE

    python -m llava.eval.model_cot_loader \
        --model-path ./checkpoints/$CKPT \
        --question-file $DATADIR/benchmark/$QUESTION_FILE.json \
        --image-folder ./playground/data/ \
        --answers-file $RESULTDIR/results/$QUESTION_FILE/$CKPT.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --with-cot true \
        --detection-file $RESULTDIR/detection/$QUESTION_FILE/$CKPT.jsonl  

done
