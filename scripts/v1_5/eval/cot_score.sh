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

for ((i=0; i<${#script_paths[@]}; i++)); do
    QUESTION_FILE=${script_paths[i]}

    mkdir -p $RESULTDIR/scores/cot/$QUESTION_FILE
    echo $RESULTDIR/results/$QUESTION_FILE/$CKPT.jsonl
    python -u llava/eval/eval_cot_score.py \
        --question-file $DATADIR/benchmark/$QUESTION_FILE.json \
        --output-result $RESULTDIR/scores/cot/$QUESTION_FILE/$CKPT.json  \
        --result-file $RESULTDIR/results/$QUESTION_FILE/$CKPT.jsonl \

done

