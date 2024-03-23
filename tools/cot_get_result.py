import json
import os
import sys
import numpy as np
from texttable import Texttable
import csv

CKPT = sys.argv[1]

datasets = ["docvqa" ,"textcap" ,"textvqa", 'dude', 'sroie',"infographicsvqa" , "flickr30k", "gqa" ,"openimages" ,"vsr", "cub"]

dataset_style_map = {
    "docvqa": "Text",
    "textcap": "Text",
    "textvqa": "Text",
    "dude": "Text",
    "sroie": "Text",
    "infographicsvqa": "Chart",
    "flickr30k": "General VQA",
    "gqa": "Reasoning VQA",
    "openimages": "Reasoning VQA",
    "vsr": "Reasoning VQA",
    "cub": "Fine-Grained",
}
dataset_styles = ["Text", "Chart", "General VQA", "Reasoning VQA", "Fine-Grained"]

resultdir = "./results/viscot/"

benchmarks = ['cot', 'cot_gtbbox', 'cot_direct', 'cot_random', 'cot_center', 'cot_woimg']


rows = []
row = []
row.append('Benchmark/Dataset')
row.extend([x.capitalize() for x in datasets])
row.append('Average')
rows.append(row)

general_rows = []
general_row = []
general_row.append('Benchmark/Dataset Style')
general_row.extend([x.capitalize() for x in dataset_styles])
general_row.append('Average')
general_rows.append(general_row)

for benchmark in benchmarks:
    row = []
    row.append(benchmark.capitalize())
    general_row = []
    general_row.append(benchmark.capitalize())

    style_score_dict = {}
    for dataset_style in dataset_styles:
        style_score_dict[dataset_style] = []

    score_dict = {}
    for dataset in datasets:
        scores = []
        if not os.path.exists(os.path.join(resultdir, 'scores', benchmark, dataset, CKPT+'.json')):
            scores = [0.0]
        else:
            data = json.load(open(os.path.join(resultdir, 'scores', benchmark, dataset, CKPT+'.json')))
            for item in data:
                scores.append(item['score'])
        score_dict[dataset] = np.mean(scores)
        style_score_dict[dataset_style_map[dataset]].append(np.mean(scores))

        row.append("%.3f" % np.mean(scores))
    row.append("%.3f" % np.mean(list(score_dict.values())))
    rows.append(row)

    for dataset_style in dataset_styles:
        general_row.append("%.3f" % np.mean(style_score_dict[dataset_style]))
    general_row.append("%.3f" % np.mean(list(score_dict.values())))
    general_rows.append(general_row)

t = Texttable(max_width=160)
t.add_rows(rows)
for row in rows:
    print(' & '.join(row))
print(t.draw())

t = Texttable(max_width=160)
t.add_rows(general_rows)
print(t.draw())

# writer = csv.writer(open('scores_csv/'+CKPT+'.csv', 'w'))
# for row in rows:
#     writer.writerow(row)
