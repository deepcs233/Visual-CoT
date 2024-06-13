import json
import random
import os
import sys
import copy

from tqdm import tqdm

def get_bbox_str(bboxs, width, height):
    if len(bboxs) > 1:
        large_bbox = []
        large_bbox.append(min([x[0] for x in bboxs]))
        large_bbox.append(min([x[1] for x in bboxs]))
        large_bbox.append(max([x[2] for x in bboxs]))
        large_bbox.append(max([x[3] for x in bboxs]))
        bbox = large_bbox
    else:
        bbox = bboxs[0]
    if width > height:
        bbox[1] += (width - height) // 2
        bbox[3] += (width - height) // 2
        bbox = [x/width for x in bbox]
    else:
        bbox[0] += (height - width) // 2
        bbox[2] += (height - width) // 2
        bbox = [x/height for x in bbox]
    return '[%0.3f, %0.3f, %0.3f, %0.3f]' % (bbox[0], bbox[1], bbox[2], bbox[3])

def get_bbox_str2(bboxs):
    if len(bboxs) > 1:
        large_bbox = []
        large_bbox.append(min([x[0] for x in bboxs]))
        large_bbox.append(min([x[1] for x in bboxs]))
        large_bbox.append(max([x[2] for x in bboxs]))
        large_bbox.append(max([x[3] for x in bboxs]))
        bbox = large_bbox
    else:
        bbox = bboxs[0]
    return '[%d, %d, %d, %d]' % (bbox[0], bbox[1], bbox[2], bbox[3])

base_cot_prompt = ' Please provide the bounding box coordinate of the region that can help you answer the question better.'

advanced_cot_prompt = ' Please think step by step and provide the bounding box coordinate of the region that can help you answer the question better.'


if __name__ == '__main__':
    file_path = sys.argv[1]
    save_path = sys.argv[2]

    lines = open(file_path).readlines()
    data = []
    for index, line in tqdm(enumerate(lines)):
        item = json.loads(line)
        bboxs = item['bboxs']
        if len(bboxs) == 0:
            print(item)
        new_item = {}
        new_item['dataset'] = item['dataset']
        new_item['split'] = item['split']
        new_item['question_id'] = index
        new_item['image'] = []
        if item['dataset'] == 'textcap':
            new_item['image'].append(os.path.join('cot', 'textvqa', item['image']))
            new_item['image'].append(os.path.join('cot', 'textvqa', item['image'])+'###'+get_bbox_str2(bboxs))
        else:
            new_item['image'].append(os.path.join('cot', item['dataset'], item['image']))
            new_item['image'].append(os.path.join('cot', item['dataset'], item['image'])+'###'+get_bbox_str2(bboxs))
        bbox_str = get_bbox_str(bboxs, item['width'], item['height'])
        conversations = []
        conversations.append({'from': 'human', 'value': '<image>\n' + item['question'] + base_cot_prompt})
        conversations.append({'from': 'gpt', 'value': bbox_str})
        conversations.append({'from': 'human', 'value': '<image>'})
        conversations.append({'from': 'gpt', 'value': item['answer']})
        new_item['conversations'] = conversations

        data.append(new_item)
    json.dump(data, open(save_path, 'w'), indent=4)

