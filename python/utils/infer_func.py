import torch
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from axengine import InferenceSession
import os
import re
from ml_dtypes import bfloat16


# Discover model files automatically from model_dir.
# We expect files like: <prefix>_p128_l<idx>_together.axmodel and <prefix>_post.axmodel
# we try to detect model prefix and layer files automatically
def _find_axmodel_files(base_dir: str, expected_layers: int = None, expected_prefill: int = 128):
    files = os.listdir(base_dir)
    # match prefix, prefill size (dynamic), and layer index
    layer_pattern = re.compile(r"^(?P<prefix>.*)_p(?P<prefill>\d+)_l(?P<idx>\d+)_together\.axmodel$")
    post_pattern = re.compile(r"^(?P<prefix>.*)_post\.axmodel$")

    # collect prefix -> [(idx, fname)]
    prefix_map = {}
    for fname in files:
        m = layer_pattern.match(fname)
        if m:
            prefix = m.group("prefix")
            idx = int(m.group("idx"))
            prefix_map.setdefault(prefix, []).append((idx, fname))

    if not prefix_map:
        # fallback to hardcoded pattern if nothing detected
        prefix = "gemma3_text"
        layer_files = [(
            i, f"{prefix}_p{expected_prefill}_l{i}_together.axmodel"
        ) for i in range(expected_layers or 0)]
    else:
        # choose the prefix with the most layers (most likely the correct one)
        prefix = max(prefix_map.items(), key=lambda kv: len(kv[1]))[0]
        # debug info
        print(f"Detected prefixes: {list(prefix_map.keys())}, chosen: {prefix}, layers: {len(prefix_map[prefix])}")
        layer_files = sorted(prefix_map[prefix], key=lambda it: it[0])

    # find post process file
    post_file = None
    for fname in files:
        m = post_pattern.match(fname)
        if m and m.group("prefix") == prefix:
            post_file = fname
            break
    if post_file is None:
        candidate = os.path.join(base_dir, f"{prefix}_post.axmodel")
        if os.path.exists(candidate):
            post_file = f"{prefix}_post.axmodel"
        else:
            for fname in files:
                if fname.endswith("_post.axmodel"):
                    post_file = fname
                    break

    return layer_files, post_file, prefix

class InferManager:
    def __init__(self, config, model_dir, max_seq_len=2047):

        self.config = config
        self.max_seq_len = max_seq_len

        self.sub_dim = config.hidden_size // config.num_attention_heads if not config.head_dim else config.head_dim
        self.kv_dim = self.sub_dim * config.num_key_value_heads

        self.k_caches = [
            np.zeros((1, self.max_seq_len, self.kv_dim), dtype=bfloat16)
            for _ in range(config.num_hidden_layers)
        ]
        self.v_caches = [
            np.zeros((1, self.max_seq_len, self.kv_dim), dtype=bfloat16)
            for _ in range(config.num_hidden_layers)
        ]

        layer_files, post_file, prefix = _find_axmodel_files(model_dir, config.num_hidden_layers)

        self.decoder_sessions = []
        for _, fname in tqdm(layer_files, desc="Init InferenceSession"):
            session = InferenceSession(os.path.join(model_dir, fname))
            self.decoder_sessions.append(session)

        # post_file was returned by _find_axmodel_files; ensure it was found
        if post_file is None:
            raise FileNotFoundError("Cannot find post process .axmodel file in model_dir")
        self.post_process_session = InferenceSession(os.path.join(model_dir, post_file))
        print("Model loaded successfully!")

    @staticmethod
    def _top_p(probs: np.ndarray, p: float) -> np.ndarray:
        sorted_indices = np.argsort(probs)
        filtered = probs.copy()
        cumulative = 0
        for idx in sorted_indices[::-1]:
            if cumulative >= p:
                filtered[idx] = 0
            cumulative += filtered[idx]
        return filtered / cumulative

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max()
        exp_logits = np.exp(logits)
        return (exp_logits / np.sum(exp_logits)).astype(np.float64)

    def post_process(self, logits, top_k=1, top_p=0.9, temperature=0.6):
        logits = logits.astype(np.float32).flatten()
        candidate_indices = np.argpartition(logits, -top_k)[-top_k:]
        candidate_logits = logits[candidate_indices] / temperature
        candidate_probs = self._softmax(candidate_logits)
        candidate_probs = self._top_p(candidate_probs, top_p)
        candidate_probs = candidate_probs.astype(np.float64) / candidate_probs.sum()
        chosen_idx = np.random.multinomial(1, candidate_probs).argmax()
        next_token = candidate_indices[chosen_idx]
        return next_token, candidate_indices, candidate_probs

    def gen_slice_indices(self, token_len, prefill=128, expand=128):
        remaining = max(0, token_len - prefill)
        extra_blocks = (remaining + expand - 1) // expand
        return list(range(extra_blocks + 1))

    def prefill(
        self,
        tokenizer,
        token_ids,
        embed_data,
        slice_len=128,
    ):
        """
        Prefill step for chunked inference.
        """
        seq_len = len(token_ids)
        slice_indices = [i for i in range(seq_len // slice_len + 1)]
        print(f"slice_indices: {slice_indices}")
        # total_prefill_len = (
        #     slice_len * slice_indices[-1]
        #     if slice_indices[-1] != 0
        #     else slice_len
        # )
        total_prefill_len = slice_len * (slice_indices[-1] + 1)
        # slice_indices = self.gen_slice_indices(seq_len)

        if total_prefill_len > 0:
            for slice_idx in slice_indices:
                indices = np.arange(
                    slice_idx * slice_len,
                    (slice_idx + 1) * slice_len,
                    dtype=np.uint32
                ).reshape((1, slice_len))

                mask = (
                    np.zeros((1, slice_len, slice_len * (slice_idx + 1)))
                    - 65536
                )
                data = np.zeros((1, slice_len, self.config.hidden_size)).astype(bfloat16)
                for i, t in enumerate(
                    range(
                        slice_idx * slice_len,
                        (slice_idx + 1) * slice_len,
                    )
                ):
                    if t < len(token_ids):
                        mask[:, i, : slice_idx * slice_len + i + 1] = 0
                        data[:, i : i + 1, :] = (
                            embed_data[t]
                            .reshape((1, 1, self.config.hidden_size))
                            .astype(bfloat16)
                        )

                remain_len = (
                    seq_len - slice_idx * slice_len
                    if slice_idx == slice_indices[-1]
                    else slice_len
                )
                mask = mask.astype(bfloat16)
                for layer_idx in range(self.config.num_hidden_layers):
                    input_feed = {
                        "K_cache": (
                            self.k_caches[layer_idx][:, 0 : slice_len * slice_idx, :]
                            if slice_idx
                            else np.zeros((1, 1, self.config.hidden_size), dtype=bfloat16)
                        ),
                        "V_cache": (
                            self.v_caches[layer_idx][:, 0 : slice_len * slice_idx, :]
                            if slice_idx
                            else np.zeros((1, 1, self.config.hidden_size), dtype=bfloat16)
                        ),
                        "indices": indices,
                        "input": data,
                        "mask": mask,
                    }
                    outputs = self.decoder_sessions[layer_idx].run(None, input_feed, shape_group=slice_idx + 1)
                    self.k_caches[layer_idx][
                        :,
                        slice_idx * slice_len : slice_idx * slice_len + remain_len,
                        :,
                    ] = outputs[0][:, :remain_len, :]
                    self.v_caches[layer_idx][
                        :,
                        slice_idx * slice_len : slice_idx * slice_len + remain_len,
                        :,
                    ] = outputs[1][:, :remain_len, :]
                    data = outputs[2]

                print("Slice prefill done:", slice_idx)
            
            # return data[:, :remain_len, :]
            post_out = self.post_process_session.run(
                None,
                {
                    "input": data[
                        :, seq_len - (len(slice_indices) - 1) * slice_len - 1, None, :
                    ]
                }
            )[0]
            next_token, possible_tokens, possible_probs = self.post_process(post_out)
            possible_decoded = [tokenizer.decode([t]) for t in possible_tokens]
            possible_probs_str = [str((t, p)) for t, p in zip(possible_decoded, possible_probs)]
            token_ids.append(next_token)
            return token_ids

    def decode(
        self,
        tokenizer,
        token_ids,
        embed_matrix,
        prefill_len=128,
        slice_len=128,
        eos_token_id=None, # 某些模型有多个 eos_token_id
        stream=True,
    ):
        """Autoregressive decode; optionally stream tokens or collect silently."""

        decoded_text = tokenizer.decode(token_ids[-1], skip_special_tokens=True)
        if stream:
            print("answer >>", decoded_text, end='', flush=True)

        mask = np.zeros((1, 1, self.max_seq_len + 1), dtype=np.float32).astype(bfloat16)
        mask[:, :, :self.max_seq_len] -= 65536
        seq_len = len(token_ids) - 1
        if prefill_len > 0:
            mask[:, :, :seq_len] = 0

        for step_idx in range(self.max_seq_len):
            if prefill_len > 0 and step_idx < seq_len:
                continue
            cur_token = token_ids[step_idx]
            indices = np.array([step_idx], np.uint32).reshape((1, 1))
            data = embed_matrix[cur_token, :].reshape((1, 1, self.config.hidden_size)).astype(bfloat16)
            for layer_idx in range(self.config.num_hidden_layers):
                input_feed = {
                    "K_cache": self.k_caches[layer_idx],
                    "V_cache": self.v_caches[layer_idx],
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }
                outputs = self.decoder_sessions[layer_idx].run(None, input_feed, shape_group=0)
                self.k_caches[layer_idx][:, step_idx, :] = outputs[0][:, :, :]
                self.v_caches[layer_idx][:, step_idx, :] = outputs[1][:, :, :]
                data = outputs[2]
            mask[..., step_idx] = 0
            if step_idx < seq_len - 1:
                continue
            else:
                post_out = self.post_process_session.run(None, {"input": data})[0]
                next_token, possible_tokens, possible_probs = self.post_process(post_out, temperature=0.7)
                if eos_token_id is not None and next_token in eos_token_id:
                    break
                elif next_token == tokenizer.eos_token_id:
                    break
                token_ids.append(next_token)

            decoded_piece = tokenizer.decode(next_token, skip_special_tokens=True)
            decoded_text += decoded_piece
            if stream:
                print(decoded_piece, end='', flush=True)

        return decoded_text

