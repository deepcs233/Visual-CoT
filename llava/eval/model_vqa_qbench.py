import argparse
import json
from io import BytesIO
import os 

import requests
import torch
from PIL import Image
from tqdm import tqdm

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
    load_image_from_base64,
    get_model_name_from_path,
    KeywordsStoppingCriteria
)


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name)#, True)

    with open(args.questions_file) as f:
        llvqa_data = json.load(f)

    for i, llddata in enumerate(tqdm(llvqa_data)):
        image_file = llddata["img_path"]
        if args.lang == "en":
            message = llddata["question"] + \
                "\nChoose between one of the options as follows:\n"
        elif args.lang == "zh":
            message = llddata["question"] + "\在下列选项中选择一个:\n"
        else:
            raise NotImplementedError(
                "Q-Bench does not support languages other than English (en) and Chinese (zh) yet. Contact us (https://github.com/VQAssessment/Q-Bench/) to convert  Q-Bench into more languages.")
        for choice, ans in zip(["A.", "B.", "C.", "D."], llddata["candidates"]):
            message += f"{choice} {ans}\n"
        qs = message
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
                DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        
        # if 'llama-2' in model_name.lower():
        #     conv_mode = "share4v_llama_2"
        # elif "v1" in model_name.lower():
        #     conv_mode = "share4v_v1"
        # elif "mpt" in model_name.lower():
        #     conv_mode = "mpt"
        # else:
        #     conv_mode = "share4v_v0"

        # if args.conv_mode is not None and conv_mode != args.conv_mode:
        #     print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(
        #         conv_mode, args.conv_mode, args.conv_mode))
        # else:
        #     args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(args.image_folder, image_file))
        if isinstance(image_processor, list):
            image_tensor_0 = image_processor[0].preprocess(
                image, return_tensors="pt"
            )["pixel_values"][0]
            image_tensor_1 = image_processor[1].preprocess(
                image, return_tensors="pt"
            )["pixel_values"][0]
            image_tensor = torch.cat((image_tensor_0, image_tensor_1), dim=0)
            if len(image_processor)>2:
                image_tensor_2 = image_processor[2].preprocess(
                    image, return_tensors="pt"
                )["pixel_values"][0]
                high_image_tensor = image_tensor_2
            else:
                high_image_tensor = image_tensor_0
            if len(image_processor)>3:
                image_tensor_3 = image_processor[3].preprocess(
                    image, return_tensors="pt"
                )["pixel_values"][0]
                flattened_image_tensor = image_tensor_3
            else:
                flattened_image_tensor = image_tensor_0                       
        else:
            image_tensor = image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
            high_image_tensor = image_tensor
            flattened_image_tensor = image_tensor
        images = image_tensor.unsqueeze(0).bfloat16().cuda()
        high_images = high_image_tensor.unsqueeze(0).bfloat16().cuda()
        flattened_patches = flattened_image_tensor.unsqueeze(0).bfloat16().cuda()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = (
            [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)]
            if conv.version == "v0"
            else None
        )

        if "qformer" in model_name or 'gate' in model_name:
            prompts = [["question 0: <q>\n\n".replace("<q>", qs.replace("<image>\n", "").lower())]]
        else:
            prompts = None

        if "gate-mask" in model_name:
            routing_weights = torch.Tensor([1, 0, 1, 0, 0, 0, 0])
            routing_weights = routing_weights.unsqueeze(0).bfloat16().cuda()
        else:
            routing_weights = None

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                high_images=high_images if isinstance(image_processor, list) and len(image_processor)>2  else None,
                flattened_patches=flattened_patches if isinstance(image_processor, list) and len(image_processor)>3  else None,
                routing_weights=routing_weights,
                prompts=prompts,              
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True,
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (
            input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        llddata["response"] = outputs
        with open(args.answers_file, "a") as wf:
            json.dump(llddata, wf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="share4v-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str,
                        default="./playground/data/qbench/images_llvisionqa")
    parser.add_argument("--questions-file", type=str,
                        default="./playground/data/qbench/llvisionqa_dev.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)    
    parser.add_argument("--regen", action="store_true", default=False)
    args = parser.parse_args()
    if os.path.exists(args.answers_file) and not args.regen:
        print("{} already exists, won't regen again.".format(args.answers_file))
    else:
        eval_model(args)
