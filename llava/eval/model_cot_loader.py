import argparse
import random
import re
import copy
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from torch.utils.data import Dataset, DataLoader
from PIL import Image



SUBIMAGE_PATTERN = r".*\#\#\#\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\]"

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(
        self,
        questions,
        image_folder,
        tokenizer,
        image_processor,
        model_config,
        model_name,
        with_cot,
        detection_results,
        random_bbox,
        center_bbox,
        without_image,
        adapt_ratio,
    ):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.model_name = model_name
        self.with_cot = with_cot
        self.detection_results = detection_results
        self.random_bbox = random_bbox
        self.center_bbox = center_bbox
        self.without_image = without_image
        self.adapt_ratio = adapt_ratio

    def __getitem__(self, index):
        line = self.questions[index]
        image_files = line["image"]
        raw_conversations = line["conversations"]

        conv = conv_templates[args.conv_mode].copy()
        if self.random_bbox:
            center = [random.random(), random.random()]
            height = random.random() * 0.5
            width = random.random() * 0.5
            random_coords = [max(0, center[0]-width), max(0, center[1]-height), min(1, center[0]+width), min(1, center[1]+height)]
            bbox_ratio = (random_coords[2] - random_coords[0]) * (random_coords[3] - random_coords[1])

        elif self.center_bbox:
            random_coords = [0.25, 0.25, 0.75, 0.75]
            bbox_ratio = (random_coords[2] - random_coords[0]) * (random_coords[3] - random_coords[1])

        elif self.detection_results is not None:
            coords = self.detection_results[index]['text'].replace(' .','').replace('[','').replace(']','').split(', ')
            coords = [float(x) for x in coords]
            bbox_ratio = (coords[2] - coords[0]) * (coords[3] - coords[1])
        else:
            bbox_ratio = 0.0

        if self.with_cot and self.without_image is False:
            conv.append_message(conv.roles[0], raw_conversations[0]['value'].split(' Please provide the bounding box coordinate of the region')[0])
            if self.random_bbox or self.center_bbox:
                conv.append_message(conv.roles[1], '[%.3f, %.3f, %.3f, %.3f]' % (random_coords[0], random_coords[1], random_coords[2], random_coords[3]))
            elif self.detection_results is None:
                conv.append_message(conv.roles[1], raw_conversations[1]['value'])
            else:
                conv.append_message(conv.roles[1], self.detection_results[index]['text'])

            # conv.append_message(conv.roles[0], raw_conversations[2]['value'])
            conv.append_message(conv.roles[0], raw_conversations[2]['value'] + '\nPlease answer the question based on the original image and local detail image.'+  raw_conversations[0]['value'].split('Please provide the bounding box coordinate of the region')[0].replace('<image>\n', ''))
            conv.append_message(conv.roles[1], None)
        elif self.with_cot and self.without_image is True:
            conv.append_message(conv.roles[0], raw_conversations[0]['value'])
            if self.random_bbox or self.center_bbox:
                conv.append_message(conv.roles[1], '[%.3f, %.3f, %.3f, %.3f]' % (random_coords[0], random_coords[1], random_coords[2], random_coords[3]))
            elif self.detection_results is None:
                conv.append_message(conv.roles[1], raw_conversations[1]['value'])
            else:
                conv.append_message(conv.roles[1], self.detection_results[index]['text'])
            conv.append_message(conv.roles[0], '')
            conv.append_message(conv.roles[1], None)
        else:
            if 'Please provide the bounding box' in raw_conversations[0]['value']:
                conv.append_message(conv.roles[0], raw_conversations[0]['value'].split('Please provide the bounding box')[0])
            else:
                conv.append_message(conv.roles[0], raw_conversations[0]['value'])
            conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images = []
        image_path = os.path.join(self.image_folder, image_files[0])
        image = Image.open(image_path).convert("RGB")
        images.append(image)


        if self.with_cot and self.without_image is False and len(image_files) > 1:
            if self.random_bbox or self.center_bbox:
                coords = random_coords
            elif self.detection_results is None:
                if '###' not in image_files[1]:
                    raise ValueError("%s is not a valid cot path" % image_path)
                try:
                    coords = raw_conversations[1]['value'].replace(' .','').replace('[','').replace(']','').split(', ')
                    coords = [float(x) for x in coords]
                except Exception as e:
                    print(e)
                    print("Can not parse the coords: %s" % image_files[1])
                    coords = [0.0, 0.0, 1.0, 1.0]
            else:
                try:
                    coords = self.detection_results[index]['text'].replace(' .','').replace('[','').replace(']','').split(', ')
                    coords = [float(x) for x in coords]
                except Exception as e:
                    print(e)
                    print("Can not parse the coords: %s" % self.detection_results[index]['text'])
                    coords = [0.0, 0.0, 1.0, 1.0]
            image_files[1] = image_files[1].split('###')[0]
            image_path2 = os.path.join(self.image_folder, image_files[1])
            if image_path2 == image_path:
                image = copy.copy(images[0])
            else:
                image = Image.open(image_path2).convert("RGB")

            def cropwithbbox(pil_img, sub_image_info):
                width, height = pil_img.size
                x_min, y_min, x_max, y_max = sub_image_info
                if sum([x_min, y_min, x_max, y_max]) < 5:
                    x_min = x_min * max(width, height)
                    y_min = y_min * max(width, height)
                    x_max = x_max * max(width, height)
                    y_max = y_max * max(width, height)
                if width > height:
                    overlay = (width - height) // 2
                    y_min = max(0, y_min - overlay)
                    y_max = max(0, y_max - overlay)
                else:
                    overlay = (height - width) // 2
                    x_min = max(0, x_min - overlay)
                    x_max = max(0, x_max - overlay)
                center_point = [(x_min + x_max)//2, (y_min + y_max)//2]
                half_sizes = [(x_max - x_min)//2, (y_max - y_min)//2]
                cropped_half_size = max(max(half_sizes), 112)
                upper_left_point = [center_point[0]-cropped_half_size, center_point[1]-cropped_half_size]
                if upper_left_point[0] < 0:
                    center_point[0] += (-upper_left_point[0])
                if upper_left_point[1] < 0:
                    center_point[1] += (-upper_left_point[1])
                lower_right_point = [center_point[0]+cropped_half_size, center_point[1]+cropped_half_size]
                if lower_right_point[0] > width:
                    center_point[0] -= (lower_right_point[0] - width)
                if lower_right_point[1] > height:
                    center_point[1] -= (lower_right_point[1] - height)
                cropped_region = [max(0, center_point[0]-cropped_half_size), max(0, center_point[1]-cropped_half_size), min(width, center_point[0]+cropped_half_size), min(height, center_point[1]+cropped_half_size)]
                cropped_image = pil_img.crop(cropped_region)
                return cropped_image
            image = cropwithbbox(image, coords)
            images.append(image)


        if isinstance(self.image_processor, list):
            image_tensor_0 = process_images(
                images, self.image_processor[0], self.model_config
            )
            image_tensor_1 = process_images(
                images, self.image_processor[1], self.model_config
            )
            image_tensor = torch.cat((image_tensor_0, image_tensor_1), dim=0)
        else:
            image_tensor = process_images(
                images, self.image_processor, self.model_config
            )

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )

        return input_ids, image_tensor, prompt

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(
    questions,
    image_folder,
    tokenizer,
    image_processor,
    model_config,
    model_name,
    with_cot,
    detection_results,
    random_bbox,
    center_bbox,
    without_image,
    adapt_ratio,
    batch_size=1,
    num_workers=4,
):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(
        questions, image_folder, tokenizer, image_processor, model_config, model_name, with_cot, detection_results, random_bbox, center_bbox, without_image, adapt_ratio
    )
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    if args.random_bbox is True and args.center_bbox is True:
        raise ValueError("random-bbox and center-bbox cannot all be true!")

    if args.question_file.endswith('.jsonl'):
        questions = [
            json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
        ]
    else:
        questions = json.load(open(args.question_file))
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if args.detection_file is not None:
        detection_results = [
            json.loads(r) for r in open(args.detection_file, 'r')
        ]
    else:
        detection_results = None

    if (
        "plain" in model_name
        and "finetune" not in model_name.lower()
        and "mmtag" not in args.conv_mode
    ):
        args.conv_mode = args.conv_mode + "_mmtag"
        print(
            f"It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}."
        )
    data_loader = create_data_loader(
        questions,
        args.image_folder,
        tokenizer,
        image_processor,
        model.config,
        model_name,
        args.with_cot,
        detection_results,
        args.random_bbox,
        args.center_bbox,
        args.without_image,
        args.adapt_ratio,
    )

    for (input_ids, image_tensor, prompt), line in tqdm(
        zip(data_loader, questions), total=len(questions)
    ):
        idx = line["question_id"]

        stop_str = (
            conv_templates[args.conv_mode].sep
            if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO
            else conv_templates[args.conv_mode].sep2
        )
        input_ids = input_ids.to(device="cuda", non_blocking=True)
        if image_tensor.ndim == 5:
            image_tensor = image_tensor[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(
                    dtype=torch.bfloat16, device="cuda", non_blocking=True
                ),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=128,
                use_cache=True,
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (
            (input_ids != output_ids[:, :input_token_len]).sum().item()
        )
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        prompt_q = line['conversations'][0]['value']
        if prompt_q.startswith('<image>\n'):
            prompt_q = prompt_q.replace('<image>\n', '')
        if 'Please provide the bounding box coordinate of the region' in prompt_q:
            prompt_q = prompt_q.split('Please provide the bounding box coordinate of the region')[0]
        #print(outputs, line['conversations'][1]['value'])
        dumped_dict = {
                    "question_id": idx,
                    "conversations": prompt[0],
                    "text": outputs,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "prompt": prompt_q,
                    "metadata": {},
                }
        if 'height' in line:
            dumped_dict['height'] = line['height']
        if 'width' in line:
            dumped_dict['width'] = line['width']
        if 'bbox' in line:
            dumped_dict['bbox'] = line['bbox']
        ans_file.write(
            json.dumps(dumped_dict)
            + "\n"
        )
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="s3://mmdata/")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument('--with-cot', type=bool, default=False)
    parser.add_argument('--random-bbox', type=bool, default=False)
    parser.add_argument('--center-bbox', type=bool, default=False)
    parser.add_argument('--without-image', type=bool, default=False)
    parser.add_argument('--detection-file', type=str, default=None)
    parser.add_argument('--adapt-ratio', type=float, default=1.0)
    args = parser.parse_args()
    eval_model(args)
