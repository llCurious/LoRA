import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str)

args = parser.parse_args()

if args.task_name == "gpt2":
    cmd = f"python -m quant.convert_ft_to_int8 \
    --model_name gpt2 \
    --model_path ~/.cache/huggingface/transformers/gpt2 \
    --output_dir tmp \
    --dataset_name wikitext"
elif args.task_name == "opt":
    cmd = f"python -m quant.convert_ft_to_int8 \
    --model_name opt \
    --model_path ~/.cache/huggingface/transformers/opt-1.3b \
    --output_dir tmp \
    --dataset_name wikitext"
else:
    raise NotImplementedError(f"Task: {args.task_name} is not supported.")

print(f"CMD: {cmd}")
subprocess.run(cmd, shell=True)