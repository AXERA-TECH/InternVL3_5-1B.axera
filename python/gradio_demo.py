import argparse
import os
import time
from typing import Any, Dict, List, Optional, Generator, Tuple

import gradio as gr
import numpy as np
import torch
import torchvision.transforms as T
from ml_dtypes import bfloat16
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoTokenizer

from utils.infer_func import InferManager
from axengine import InferenceSession

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_PLACEHOLDER_TOKEN_ID = 151669  # <img>
IMG_CONTEXT_REPEAT = 256  # number of image context tokens expected by the model


SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型, 英文名叫 InternVL3, "
    "是一个有用无害的人工智能助手, 擅长思考和回答用户的问题. 请你在回答问题时使用简体中文."
    "<|im_end|>\n"
)


def build_transform(input_size: int):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform


def dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 12, image_size: int = 448,
                       use_thumbnail: bool = False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    def find_closest_aspect_ratio(ar: float, ratios: List[tuple]):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = orig_width * orig_height
        for ratio in ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(ar - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_file: Image.Image, input_size: int = 448, max_num: int = 12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image_file, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternVLGradioDemo:
    def __init__(self, hf_model: str, axmodel_dir: str, vit_axmodel: str, max_seq_len: int = 2047):
        self.hf_model = hf_model
        self.axmodel_dir = axmodel_dir
        self.vit_axmodel = vit_axmodel
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embeds = np.load(os.path.join(axmodel_dir, "model.embed_tokens.weight.npy"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
        config = AutoConfig.from_pretrained(self.hf_model, trust_remote_code=True)
        self.cfg = config.llm_config

        self.vit_session = InferenceSession(self.vit_axmodel)
        self.infer_manager = InferManager(self.cfg, self.axmodel_dir, max_seq_len=max_seq_len)

    def _build_single_turn_prompt(self, user_text: str, vit_features: List[np.ndarray]):
        prompt = SYSTEM_PROMPT
        prompt += f"<|im_start|>user\n{user_text}"
        for _ in vit_features:
            prompt += "\n<img>" + "<IMG_CONTEXT>" * IMG_CONTEXT_REPEAT + "</img>"
        prompt += "<|im_end|>\n<|im_start|>assistant\n"
        return prompt

    def _insert_vision_features(self, token_ids: List[int], prefill_data: np.ndarray, vit_features: List[np.ndarray]):
        image_start_indices = np.where(np.array(token_ids) == IMG_PLACEHOLDER_TOKEN_ID)[0].tolist()
        if len(image_start_indices) != len(vit_features):
            raise ValueError("图片数量与占位符数量不一致, 请检查输入和模板生成逻辑")
        for idx, image_start_index in enumerate(image_start_indices):
            insert_pos = image_start_index + 1
            prefill_data[insert_pos: insert_pos + IMG_CONTEXT_REPEAT] = vit_features[idx][0, :, :]
        return prefill_data

    def _run_model(self, prompt: str, vit_features: List[np.ndarray]):
        """Non-streaming推理，保留以防需要一次性结果。"""
        for k_cache in self.infer_manager.k_caches:
            k_cache.fill(0)
        for v_cache in self.infer_manager.v_caches:
            v_cache.fill(0)

        token_ids = self.tokenizer.encode(prompt)
        prefill_data = np.take(self.embeds, token_ids, axis=0).astype(bfloat16)
        if vit_features:
            prefill_data = self._insert_vision_features(token_ids, prefill_data, vit_features)

        eos_token_id = None
        if isinstance(self.cfg.eos_token_id, list) and len(self.cfg.eos_token_id) > 1:
            eos_token_id = self.cfg.eos_token_id

        slice_len = 128
        token_ids = self.infer_manager.prefill(self.tokenizer, token_ids, prefill_data, slice_len=slice_len)
        return self.infer_manager.decode(
            self.tokenizer,
            token_ids,
            self.embeds,
            slice_len=slice_len,
            eos_token_id=eos_token_id,
            stream=False,
        )

    def _stream_generate(self, prompt: str, vit_features: List[np.ndarray]):
        """流式生成，逐 token 产出累积文本与计时信息 (TTFT 与平均 decode ms/token)。"""
        # reset kv cache per request
        for k_cache in self.infer_manager.k_caches:
            k_cache.fill(0)
        for v_cache in self.infer_manager.v_caches:
            v_cache.fill(0)

        token_ids = self.tokenizer.encode(prompt)
        prefill_data = np.take(self.embeds, token_ids, axis=0).astype(bfloat16)
        if vit_features:
            prefill_data = self._insert_vision_features(token_ids, prefill_data, vit_features)

        eos_token_id = None
        if isinstance(self.cfg.eos_token_id, list) and len(self.cfg.eos_token_id) > 1:
            eos_token_id = self.cfg.eos_token_id

        slice_len = 128
        t_start = time.time()
        token_ids = self.infer_manager.prefill(self.tokenizer, token_ids, prefill_data, slice_len=slice_len)

        # copy decode逻辑，实现手动流式输出
        mask = np.zeros((1, 1, self.infer_manager.max_seq_len + 1), dtype=np.float32).astype(bfloat16)
        mask[:, :, :self.infer_manager.max_seq_len] -= 65536
        seq_len = len(token_ids) - 1
        if slice_len > 0:
            mask[:, :, :seq_len] = 0

        generated_text = self.tokenizer.decode(token_ids[-1], skip_special_tokens=True)
        ttft_ms: Optional[float] = None
        decode_tokens = 0
        decode_elapsed_ms: float = 0.0
        yield generated_text, ttft_ms, None, None, False

        for step_idx in range(self.infer_manager.max_seq_len):
            if slice_len > 0 and step_idx < seq_len:
                continue
            cur_token = token_ids[step_idx]
            indices = np.array([step_idx], np.uint32).reshape((1, 1))
            data = self.embeds[cur_token, :].reshape((1, 1, self.cfg.hidden_size)).astype(bfloat16)
            for layer_idx in range(self.cfg.num_hidden_layers):
                input_feed = {
                    "K_cache": self.infer_manager.k_caches[layer_idx],
                    "V_cache": self.infer_manager.v_caches[layer_idx],
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }
                outputs = self.infer_manager.decoder_sessions[layer_idx].run(None, input_feed, shape_group=0)
                self.infer_manager.k_caches[layer_idx][:, step_idx, :] = outputs[0][:, :, :]
                self.infer_manager.v_caches[layer_idx][:, step_idx, :] = outputs[1][:, :, :]
                data = outputs[2]
            mask[..., step_idx] = 0
            if step_idx < seq_len - 1:
                continue
            post_out = self.infer_manager.post_process_session.run(None, {"input": data})[0]
            next_token, possible_tokens, possible_probs = self.infer_manager.post_process(post_out, temperature=0.7)
            if eos_token_id is not None and next_token in eos_token_id:
                ttft_ms = ttft_ms or (time.time() - t_start) * 1000
                break
            if next_token == self.tokenizer.eos_token_id:
                ttft_ms = ttft_ms or (time.time() - t_start) * 1000
                break

            token_ids.append(next_token)
            decoded_piece = self.tokenizer.decode(next_token, skip_special_tokens=True)
            generated_text += decoded_piece

            if ttft_ms is None:
                ttft_ms = (time.time() - t_start) * 1000
            else:
                decode_tokens += 1
                decode_elapsed_ms = (time.time() - t_start) * 1000 - ttft_ms

            avg_decode = (decode_elapsed_ms / decode_tokens) if decode_tokens > 0 else None
            yield generated_text, ttft_ms, avg_decode, decode_tokens, False

        total_ms = (time.time() - t_start) * 1000
        avg_decode = (decode_elapsed_ms / decode_tokens) if decode_tokens > 0 else None
        yield generated_text, ttft_ms, avg_decode, decode_tokens, True

    def chat(self, user_input: str, image: Optional[Image.Image]) -> Generator:
        user_text = (user_input or "").strip()
        if not user_text and image is None:
            yield [], gr.update(), gr.update(), gr.update(), gr.update()
            return

        # 先展示占位，保持图片不清空；同时占位速度信息
        yield [(user_text, "处理中…")], gr.update(value=""), gr.update(), gr.update(value="<div style='text-align: right; font-size: 13px; color: #6b7280; font-family: monospace;'>TTFT -- ms&nbsp;&nbsp;|&nbsp;&nbsp;Decode -- ms/token&nbsp;&nbsp;|&nbsp;&nbsp;Tokens --</div>"), gr.update(interactive=False)

        vit_outputs = []
        if image is not None:
            pixel_values = load_image(image, input_size=448, max_num=1)
            vit_output = self.vit_session.run(None, {"image": pixel_values.numpy()})[0]
            vit_outputs.append(vit_output.copy())

        prompt = self._build_single_turn_prompt(user_text, vit_outputs)

        chatbot_history = [(user_text, "")]  # 将在流式过程中填充
        for partial, ttft_ms, avg_decode_ms, decode_tokens, finished in self._stream_generate(prompt, vit_outputs):
            chatbot_history[-1] = (user_text, partial)
            ttft_disp = f"{ttft_ms:.0f}" if ttft_ms is not None else "--"
            decode_disp = f"{avg_decode_ms:.1f}" if avg_decode_ms is not None else "--"
            tok_disp = f"{decode_tokens}" if decode_tokens is not None else "--"
            metrics_text = f"<div style='text-align: right; font-size: 13px; color: #6b7280; font-family: monospace;'>TTFT {ttft_disp} ms&nbsp;&nbsp;|&nbsp;&nbsp;Decode {decode_disp} ms/token&nbsp;&nbsp;|&nbsp;&nbsp;Tokens {tok_disp}</div>"
            if finished:
                yield chatbot_history, gr.update(value=""), gr.update(), gr.update(value=metrics_text), gr.update(interactive=True)
            else:
                yield chatbot_history, gr.update(value=""), gr.update(), gr.update(value=metrics_text), gr.update(interactive=False)

    @staticmethod
    def build_ui(demo: "InternVLGradioDemo", server_name: str = "0.0.0.0", server_port: int = 7860, share: bool = False):
        with gr.Blocks(title="InternVL3-5-1B AX Gradio Demo", theme=gr.themes.Soft()) as iface:
            gr.HTML("""<style>
            #image-pane img {object-fit: contain; max-height: 380px;}
            #chat-wrap {position: relative;}
            #metrics-display {position: absolute; right: 12px; bottom: 12px; z-index: 5; pointer-events: none; text-align: right;}
            #metrics-display > div {display: inline-block;}
            </style>""")
            gr.Markdown("""### InternVL3-5-1B 图文对话演示\n上传一张图片 (可选)，输入问题，获取中文回答。""")

            with gr.Row():
                # 左侧：对话框和输入区域
                with gr.Column(scale=5):
                    with gr.Group(elem_id="chat-wrap"):
                        chatbot = gr.Chatbot(height=500, label="对话")
                        metrics_md = gr.Markdown("<div style='text-align: right; font-size: 13px; color: #6b7280; font-family: monospace;'>TTFT -- ms&nbsp;&nbsp;|&nbsp;&nbsp;Decode -- ms/token&nbsp;&nbsp;|&nbsp;&nbsp;Tokens --</div>", elem_id="metrics-display")

                    with gr.Row():
                        user_input = gr.Textbox(
                            placeholder="按 Enter 发送，Shift+Enter 换行",
                            lines=2,
                            scale=7,
                            max_lines=5,
                            show_label=False,
                        )
                        with gr.Column(scale=1, min_width=100):
                            send_btn = gr.Button("发送", variant="primary", size="sm")
                            clear_btn = gr.Button("清空对话", variant="secondary", size="sm")

                # 右侧：图像上传和信息提示
                with gr.Column(scale=3):
                    image_input = gr.Image(
                        type="pil",
                        label="上传图片 (可选)",
                        height=380,
                        image_mode="RGB",
                        show_download_button=False,
                        elem_id="image-pane",
                    )
                    gr.Markdown("""- 支持单张图像理解\n- 仅当前问题与回答，不保留历史\n- 处理时间取决于硬件，请耐心等待""")

            def _clear():
                return [], gr.update(value=""), gr.update(), gr.update(value="<div style='text-align: right; font-size: 13px; color: #6b7280; font-family: monospace;'>TTFT -- ms&nbsp;&nbsp;|&nbsp;&nbsp;Decode -- ms/token&nbsp;&nbsp;|&nbsp;&nbsp;Tokens --</div>"), gr.update(interactive=True)

            send_btn.click(
                fn=demo.chat,
                inputs=[user_input, image_input],
                outputs=[chatbot, user_input, image_input, metrics_md, send_btn],
                show_progress=False,
                queue=True,
            )
            user_input.submit(
                fn=demo.chat,
                inputs=[user_input, image_input],
                outputs=[chatbot, user_input, image_input, metrics_md, send_btn],
                show_progress=False,
                queue=True,
            )
            clear_btn.click(fn=_clear, inputs=None, outputs=[chatbot, user_input, image_input, metrics_md, send_btn])

        iface.queue().launch(server_name=server_name, server_port=server_port, share=share)


def parse_args():
    parser = argparse.ArgumentParser(description="InternVL3-5-1B AX gradio demo")
    parser.add_argument("--hf_model", type=str, default="./InternVL3_5-1B",
                        help="HuggingFace 模型路径")
    parser.add_argument("--axmodel_path", type=str, default="./InternVL3_5-1B_axmodel",
                        help="LLM axmodel 目录")
    parser.add_argument("--vit_model", type=str, default="./vit-models/internvl_vit_model_1x3x448x448.axmodel",
                        help="ViT axmodel 路径")
    parser.add_argument("--port", type=int, default=7860, help="Gradio 端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Gradio 监听地址")
    parser.add_argument("--share", action="store_true", help="启用 gradio share")
    return parser.parse_args()


def main():
    args = parse_args()
    demo = InternVLGradioDemo(args.hf_model, args.axmodel_path, args.vit_model)
    InternVLGradioDemo.build_ui(demo, server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
