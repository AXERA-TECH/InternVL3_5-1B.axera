from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import onnx
import onnxruntime as ort
import numpy as np
import os
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from typing import List, Tuple
from axengine import InferenceSession
from ml_dtypes import bfloat16
from utils.infer_func import InferManager
import argparse
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

if __name__ == "__main__":

    """
    python3 infer_axmodel.py  --vit_model vit-models/internvl_vit_model_1x3x448x448.axmodel --images examples/image_0.jpg
    """
    prompt = None
    parser = argparse.ArgumentParser(description="Model configuration parameters")
    parser.add_argument("--hf_model", type=str, default="./InternVL3_5-1B",
                        help="Path to HuggingFace model")
    parser.add_argument("--axmodel_path", type=str, default="./InternVL3_5-1B_axmodel",
                        help="Path to save compiled axmodel of llama model")
    parser.add_argument("--vit_model", type=str, default=None, help="Path to save compiled axmodel of llama model")
    parser.add_argument("-i", "--images", nargs='+', type=str, default=None,
                        help="Path to the test image.")
    parser.add_argument("-q", "--question", type=str, default="请你描述这幅图的内容.",
                        help="Your question that you want to ask the model.")
    args = parser.parse_args()

    hf_model_path = args.hf_model
    axmodel_path = args.axmodel_path
    images = args.images
    prompt = args.question

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeds = np.load(os.path.join(axmodel_path, "model.embed_tokens.weight.npy"))

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)

    # model = AutoModelForCausalLM.from_pretrained(
    #     hf_model_path,
    # ).to(device)

    test_imgs_path = args.images
    vit_axmodel_path = args.vit_model

    # set the max number of tiles in `max_num`
    pixel_values_list = []
    if test_imgs_path is not None:
        for img_path in test_imgs_path:
            pixel_values = load_image(img_path, input_size=448, max_num=1)
            pixel_values_list.append(pixel_values)
        print(f"输入图像数: {len(pixel_values_list)}")
        print("preprocess image done!")

        # extract img feature by vit
        vit_session = InferenceSession(vit_axmodel_path)
        vit_output_list = []
        for idx, pixel_values in enumerate(pixel_values_list):
            vit_output = vit_session.run(None, {"image": pixel_values.numpy()})[0]
            vit_output_list.append(vit_output.copy()) # 避免 vit 输出结果使用同一块内存

        print(f"vit_output.shape is {vit_output_list[0].shape}, vit feature extract done!")

    prompt = "<|im_start|>system\n你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型, 英文名叫 InternVL3, 是一个有用无害的人工智能助手, 擅长思考和回答用户的问题. 请你在回答问题时使用简体中文.<|im_end|>\n"
    question = args.question
    prompt += "<|im_start|>user\n" + question

    if len(pixel_values_list) > 0:
        for idx in range(len(pixel_values_list)):
            prompt += "\n<img>" + "<IMG_CONTEXT>" * 256 + "</img>\n"
    prompt += "<|im_end|>\n<|im_start|>assistant\n"
    print(f"prompt is {prompt}")
    token_ids = tokenizer.encode(prompt)
    # 图像理解
    image_start_indices = np.where(np.array(token_ids) == 151669)[0].tolist() # <img> tag 151669, 151665
    prefill_data = np.take(embeds, token_ids, axis=0)
    prefill_data = prefill_data.astype(bfloat16)
    token_len = len(token_ids)

    for idx, image_start_index in enumerate(image_start_indices):
        image_insert_index = image_start_index + 1
        prefill_data[image_insert_index : image_insert_index + 256] = vit_output_list[idx][0, :, :]
    ##################################

    cfg = config.llm_config

    eos_token_id = None
    if isinstance(cfg.eos_token_id, list) and len(cfg.eos_token_id) > 1:
        eos_token_id = cfg.eos_token_id

    slice_len = 128
    prefill_max_len = 1024 - 1
    max_seq_len = 2048 - 1  # prefill + decode max length

    imer = InferManager(cfg, axmodel_path, max_seq_len=max_seq_len) # prefill + decode max length
    # import pdb; pdb.set_trace()
    token_ids = imer.prefill(tokenizer, token_ids, prefill_data, slice_len=slice_len)
    imer.decode(tokenizer, token_ids, embeds, slice_len=slice_len, eos_token_id=eos_token_id)
    print("\n")