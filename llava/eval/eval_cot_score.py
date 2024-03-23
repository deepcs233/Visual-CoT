import os
import openai
import time
import numpy as np
from tqdm import tqdm
import json
import argparse
import re
import requests
import threading


BASE_PROMPT = """
You are responsible for proofreading the answers, you need to give a score to the model's answer by referring to the standard answer, based on the given question. The full score is 1 point and the minimum score is 0 points. Please output the score in the form "score: <score>". The evaluation criteria require that the closer the model's answer is to the standard answer, the higher the score.
"""

PROMPT = """
question: %s
standard answer: %s
model's answer: %s
"""

API_KEY = ''

def make_request_openai(content, extra_args={}):
    headers = {}
    headers['Content-Type']='application/json'
    retry_times = 3
    while retry_times > 0:
        try:
            data = {}
            data['model']= "gpt-3.5-turbo-1106"
            data['messages'] = [{"role":"system","content": BASE_PROMPT}, {"role": "user", "content":content}]
            for key in extra_args:
                data[key] = extra_args[key]
            headers['Authorization'] = API_KEY
            r = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data, timeout=60)
            response = r.json()
            response = response['choices'][0]['message']['content']
            return response
        except Exception as e:
            print(e)
            time.sleep(1)
        finally:
            retry_times -= 1
    return 'unknown'


def get_score(question_text, gt_answer_text, pred_answer_text):
    content = PROMPT % (question_text, gt_answer_text, pred_answer_text)
    ret = make_request_openai(content)
    ret = ret.lower()
    if 'score' not in ret:
        return 0.0
    res = re.findall(r'score: ([\d\.]+)', ret)
    if len(res) != 1:
        return 0.0
    res = float(res[0])
    if res > 1.0:
        res = 1
    if res < 0.0:
        res = 0
    return res


def process(i):
    pred_answer = pred_answers[i]
    question_id = pred_answer['question_id']
    question = questions[question_id]

    if 'Please provide the bounding box' in question['conversations'][0]['value']:
        question_text = question['conversations'][0]['value'].split(' Please provide the bounding box')[0]
    else:
        question_text = question['conversations'][0]['value']
    gt_answer_text = question['conversations'][-1]['value']
    pred_answer_text = pred_answer['text']
    score = get_score(question_text, gt_answer_text, pred_answer_text)
    scores.append(score)
    results.append({'question_id': question_id, 'question_text': question_text, 'gt_answer_text': gt_answer_text, 'pred_answer_text': pred_answer_text, 'score': score})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument('--output-result', type=str)
    args = parser.parse_args()

    scores = []
    results = []

    if args.question_file.endswith('.jsonl'):
        questions = [json.loads(line) for line in open(args.question_file)]
        questions = {question['question_id']: question for question in questions}
    else:
        questions = json.load(open(args.question_file))
        questions = {question['question_id']: question for question in questions}
    pred_answers = [json.loads(q) for q in open(args.result_file)]

    last_num = 0
    for i in tqdm(range(len(pred_answers))):
        process(i)

    print('The avg score is: %f' % np.mean(scores))
    with open(args.output_result, 'w') as f:
        json.dump(results, f, indent=2)
