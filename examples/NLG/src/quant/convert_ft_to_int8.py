import torch
import os
import torch.nn as nn
import time

from datasets import load_dataset
import functools
from collections import defaultdict

from functools import partial
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import argparse
import datasets
from tqdm import trange

from data_utils import FT_Dataset
import transformers
import sys
sys.path.append("/home/haoqi.whq/llm-inference/smoothquant")

# from smoothquant.calibration import get_act_scales
# from smoothquant.calibration import get_static_decoder_layer_scales
# from smoothquant.opt import Int8OPTForCausalLM
from smoothquant.smooth import smooth_lm


def get_dataset(dataset_name, type: str = "validation"):
    if dataset_name == "wikitext":
        data_dir = os.path.join(
            os.path.expanduser("~"), ".cache/huggingface/datasets/wikitext"
        )
        # train_dataset = datasets.load_from_disk(data_dir)["train"]
        # eval_dataset = datasets.load_from_disk(data_dir)["validation"]
        ds = datasets.load_from_disk(data_dir)[type]
        return ds
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

def get_act_scales(model, tokenizer, dataset_name, num_samples=512, seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, transformers.pytorch_utils.Conv1D):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )

    dataset = get_dataset(dataset_name)
    dataset = dataset.shuffle(seed=42)
    print(dataset)

    for i in tqdm(range(num_samples)):
        encoded_inputs = tokenizer.encode(
            dataset[i]["text"],
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
            padding="max_length",
        ).to(device)
        label = encoded_inputs[:, 1:].to(device)

        model(encoded_inputs)

    for h in hooks:
        h.remove()

    return act_scales

@torch.no_grad()
def get_static_decoder_layer_scales(model,
                                    tokenizer,
                                    dataset_name,
                                    num_samples=512,
                                    seq_len=512,
                                    prefix="transformer.h"
                                    ):
    model.eval()
    device = next(model.parameters()).device

    act_dict = defaultdict(dict)

    def stat_io_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if name not in act_dict or "input" not in act_dict[name]:
            act_dict[name]["input"] = x.detach().abs().max().item()
        else:
            act_dict[name]["input"] = max(
                act_dict[name]["input"], x.detach().abs().max().item())
        if isinstance(y, tuple):
            y = y[0]
        if name not in act_dict or "output" not in act_dict[name]:
            act_dict[name]["output"] = y.detach().abs().max().item()
        else:
            act_dict[name]["output"] = max(
                act_dict[name]["output"], y.detach().abs().max().item())

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) or isinstance(m, transformers.pytorch_utils.Conv1D):
            hooks.append(m.register_forward_hook(
                partial(stat_io_hook, name=name)))

    print("Collecting activation scales...")
    pbar = tqdm(range(num_samples))
    dataset = get_dataset(dataset_name)
    dataset = dataset.shuffle(seed=42)
    print(dataset)
    for i in pbar:
        encoded_inputs = tokenizer.encode(
            dataset[i]["text"],
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
            padding="max_length",
        ).to(device)
        model(encoded_inputs)
        mean_scale = np.mean([v["input"] for v in act_dict.values()])
        pbar.set_description(f"Mean input scale: {mean_scale:.2f}")
    for hook in hooks:
        hook.remove()

    decoder_layer_scales = []

    for idx in range(model.config.num_hidden_layers):
        scale_dict = {}
        print(act_dict[
            f"{prefix}.{idx}.attn.c_attn"])
        scale_dict["attn_input_scale"] = act_dict[
            f"{prefix}.{idx}.attn.c_attn"]['input'] / 127
        scale_dict["attn_output_scale"] = act_dict[
            f"{prefix}.{idx}.attn.c_attn"]['output'] / 127
        # scale_dict["k_output_scale"] = act_dict[
        #     f"{prefix}.{idx}.attn.k_proj"]['output'] / 127
        # scale_dict["v_output_scale"] = act_dict[
        #     f"{prefix}.{idx}.attn.v_proj"]['output'] / 127
        scale_dict["out_input_scale"] = act_dict[
            f"{prefix}.{idx}.attn.c_proj"]['input'] / 127
        scale_dict["fc1_input_scale"] = act_dict[
            f"{prefix}.{idx}.mlp.c_fc"]['input'] / 127
        scale_dict["fc2_input_scale"] = act_dict[
            f"{prefix}.{idx}.mlp.c_proj"]["input"] / 127
        decoder_layer_scales.append(scale_dict)

    return decoder_layer_scales, act_dict


def build_model_and_tokenizer(model_name, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='gpt2', help='model name')
    parser.add_argument('--model_path', type=str,
                        default='facebook/opt-1.3b', help='model path')
    parser.add_argument('--output_dir', type=str, default='tmp',
                        help='where to save the act scales')
    parser.add_argument('--dataset_name', type=str, default='glue',
                        help='location of the calibration dataset, we use the validation set of the Pile dataset')
    parser.add_argument('--num_samples', type=int, default=512)
    parser.add_argument('--seq_len', type=int, default=512)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    print("Args:", args)

    start = time.time()
    model, tokenizer = build_model_and_tokenizer(args.model_name, args.model_path)
    print(f"{'='*10}\tBuild model and tokenizer ({(time.time() - start)}s)\t{'='*10}")


    start = time.time()
    act_scales = get_act_scales(model, tokenizer, args.dataset_name,
                                args.num_samples, args.seq_len)
    print(f"{'='*10}\tGet act scales ({(time.time() - start)}s)\t{'='*10}")
    
    
    start = time.time()
    smooth_lm(model, act_scales, 0.5)
    print(f"{'='*10}\tSmooth model ({(time.time() - start)}s)\t{'='*10}")

    start = time.time()
    act_dir = os.path.join(args.output_dir, "act_scales")
    os.makedirs(act_dir, exist_ok=True)
    act_path = os.path.join(act_dir, f"{args.model_name}.pt")
    torch.save(act_scales, act_path)
    print(f"{'='*10}\tSave act scales ({(time.time() - start)}s)\t{'='*10}")


    start = time.time()
    decoder_layer_scales, raw_scales = get_static_decoder_layer_scales(model,
                                                                       tokenizer,
                                                                       args.dataset_name,
                                                                       num_samples=args.num_samples,
                                                                       seq_len=args.seq_len)
    print(f"{'='*10}\tGet static layer scales ({(time.time() - start)}s)\t{'='*10}")

    
    weight_dir = os.path.join(args.output_dir, "weights")
    os.makedirs(weight_dir, exist_ok=True)
    weights_path = os.path.join(weight_dir, f"{args.model_name}.pt")
    torch.save(decoder_layer_scales, weights_path)

    int8_model = Int8OPTForCausalLM.from_float(model, decoder_layer_scales)
    int8_model.save_pretrained(weights_path)
    print(f"Saved int8 model at {weights_path}")

if __name__ == '__main__':
    main()
