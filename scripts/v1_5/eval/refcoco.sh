#!/bin/bash

script_paths=(
"REC_refcocog_umd_test"
"REC_refcocog_umd_val"
"REC_refcoco_unc_testA"
"REC_refcoco+_unc_testA"
"REC_refcoco_unc_testB"
"REC_refcoco+_unc_testB"
"REC_refcoco_unc_val"
"REC_refcoco+_unc_val"
)


MODEL_NAME=$1

for ((i=0; i<${#script_paths[@]}; i++)); do
    QUESTION_FILE=${script_paths[i]}

    mkdir -p ./results/multimodal/$QUESTION_FILE/results/
    python -m llava.eval.model_refcoco_loader \
        --model-path ./checkpoints/$MODEL_NAME \
        --question-file ./llava/eval/REC/$QUESTION_FILE.jsonl \
        --image-folder ./playground/data/coco/train2014 \
        --answers-file ./results/multimodal/$QUESTION_FILE/results/$MODEL_NAME.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait
