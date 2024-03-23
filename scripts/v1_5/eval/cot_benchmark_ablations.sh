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


echo START gtbbox Answer...

for ((i=0; i<${#script_paths[@]}; i++)); do
    QUESTION_FILE=${script_paths[i]}

    mkdir -p $RESULTDIR/results_gtbbox/$QUESTION_FILE

    python -m llava.eval.model_cot_loader \
        --model-path ./checkpoints/$CKPT \
        --question-file $DATADIR/benchmark/$QUESTION_FILE.json \
        --image-folder ./playground/data/ \
        --answers-file $RESULTDIR/results_gtbbox/$QUESTION_FILE/$CKPT.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --with-cot true

done



echo START direct Answer...

for ((i=0; i<${#script_paths[@]}; i++)); do
    QUESTION_FILE=${script_paths[i]}

    mkdir -p $RESULTDIR/results_direct/$QUESTION_FILE

    python -m llava.eval.model_cot_loader \
        --model-path ./checkpoints/$CKPT \
        --question-file $DATADIR/benchmark/$QUESTION_FILE.json \
        --image-folder ./playground/data/ \
        --answers-file $RESULTDIR/results_direct/$QUESTION_FILE/$CKPT.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1

done



echo START Random bbox Answer...

for ((i=0; i<${#script_paths[@]}; i++)); do
    QUESTION_FILE=${script_paths[i]}

    mkdir -p $RESULTDIR/results_random/$QUESTION_FILE

    python -m llava.eval.model_cot_loader \
        --model-path ./checkpoints/$CKPT \
        --question-file $DATADIR/benchmark/$QUESTION_FILE.json \
        --image-folder ./playground/data/ \
        --answers-file $RESULTDIR/results_random/$QUESTION_FILE/$CKPT.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --with-cot true \
        --random-bbox true

done


echo START Center bbox Answer...

for ((i=0; i<${#script_paths[@]}; i++)); do
    QUESTION_FILE=${script_paths[i]}

    mkdir -p $RESULTDIR/results_center/$QUESTION_FILE

    python -m llava.eval.model_cot_loader \
        --model-path ./checkpoints/$CKPT \
        --question-file $DATADIR/benchmark/$QUESTION_FILE.json \
        --image-folder ./playground/data/ \
        --answers-file $RESULTDIR/results_center/$QUESTION_FILE/$CKPT.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --with-cot true \
        --center-bbox true

done

