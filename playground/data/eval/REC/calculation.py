import re
import json
import os
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import torch
from torchvision.ops import box_iou

def box_xyxy_expand2square(box, *, w, h):
    if w == h:
        return box
    if w > h:
        x1, y1, x2, y2 = box
        y1 += (w - h) // 2
        y2 += (w - h) // 2
        box = x1, y1, x2, y2
        return box
    assert w < h
    x1, y1, x2, y2 = box
    x1 += (h - w) // 2
    x2 += (h - w) // 2
    box = x1, y1, x2, y2
    return box

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', default='./LaVIN', type=str)

args = parser.parse_args()
results_dir = args.results_dir
f = open(results_dir, 'r')
preds = f.readlines()

total = len(preds)
correct = 0
for i,data in enumerate(preds):
    data = json.loads(data)
    pred = data['text'].replace(' .','').replace('[','').replace(']','').split(', ')
    target = data['bbox']
    h, w = data['height'], data['width']
    target = box_xyxy_expand2square(box=target, w=w, h=h)
    # target = box
    try:
        pred = [float(pred[0])*max(w,h), float(pred[1])*max(w,h), float(pred[2])*max(w,h), float(pred[3])*max(w,h)]
        pred = torch.Tensor(pred).unsqueeze(0)
        target = torch.Tensor(target).unsqueeze(0)
        iou = box_iou(pred, target)[0]
        if iou>0.5:
            correct += 1

    except:
        print(i, data)

print(correct/total, correct, total)
