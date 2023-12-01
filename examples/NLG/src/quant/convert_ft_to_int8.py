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

# import sys
# sys.path.append("/home/haoqi.whq/llm-inference/smoothquant")
from data_utils import FT_Dataset
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel


from .opt import Int8OPTForCausalLM
from .gpt2 import Int8GPT2LMHeadModel
from .smooth import smooth_lm

from .data_utils import get_dataset
from .smooth import get_act_scales, get_static_decoder_layer_scales



def build_model_and_tokenizer(model_name, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
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
    if args.model_name.startswith("gpt2"):
        prefix = "transformer.h"
    elif args.model_name == "opt":
        prefix = "model.decoder.layers"
    decoder_layer_scales, raw_scales = get_static_decoder_layer_scales(model,
                                                                       tokenizer,
                                                                       args.dataset_name,
                                                                       num_samples=args.num_samples,
                                                                       seq_len=args.seq_len,
                                                                       prefix=prefix)
    print(f"{'='*10}\tGet static layer scales ({(time.time() - start)}s)\t{'='*10}")

    
    weight_dir = os.path.join(args.output_dir, "weights")
    os.makedirs(weight_dir, exist_ok=True)
    weights_path = os.path.join(weight_dir, f"{args.model_name}.pt")
    torch.save(decoder_layer_scales, weights_path)

    print(f"{'='*10}\tStart int8 conversion...\t{'='*10}")
    if args.model_name.startswith("gpt2"):
        int8_model = Int8GPT2LMHeadModel.from_float(model, decoder_layer_scales)
    elif args.model_name == "opt":
        int8_model = Int8OPTForCausalLM.from_float(model, decoder_layer_scales)
    int_model_path = os.path.join(args.output_dir, f"int-{args.model_name}")
    int8_model.save_pretrained(int_model_path)
    print(f"Saved int8 model at {int_model_path}")

if __name__ == '__main__':
    main()
