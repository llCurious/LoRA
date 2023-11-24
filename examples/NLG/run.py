import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str)

parser.add_argument("--lora_type", type=str, default="qkvom")
parser.add_argument("--enable_mlp_fc", action="store_true")
parser.add_argument("--enable_mlp_proj", action="store_true")
parser.add_argument("--enable_wo", action="store_true")
parser.add_argument("--enable_wq", action="store_true")
parser.add_argument("--enable_wk", action="store_true")
parser.add_argument("--enable_wv", action="store_true")

parser.add_argument("--complexity_penality", action="store_true")
parser.add_argument("--complexity_coeff", type=float, default=1e-7)

parser.add_argument("--init_checkpoint", type=str, default="./pretrained_checkpoints/gpt2-medium-pytorch_model.bin")
parser.add_argument("--lora_path", type=str)

parser.add_argument("--n_pruning", default=0, type=int)
parser.add_argument("--thr_pruning", default=0.05, type=float)
parser.add_argument("--percent_pruning", default=0, type=float)


args = parser.parse_args()

lora_type = args.lora_type
if "q" in lora_type:
    args.enable_wq = True
if "k" in lora_type:
    args.enable_wk = True
if "v" in lora_type:
    args.enable_wv = True
if "o" in lora_type:
    args.enable_wo = True
if "f" in lora_type:
    args.enable_mlp_fc = True
if "p" in lora_type:
    args.enable_mlp_proj = True
# lora_type = f"{'mlp' if args.enable_mlp else ''}_{'q' if args.enable_wq else ''}_{'k' if args.enable_wk else ''}_{'v' if args.enable_wv else ''}_{'wo' if args.enable_wo else ''}"
print(f"lora type: {lora_type}")

base_dir = os.path.join(
    "tmp",
    args.task_name,
    lora_type,
    f"cp_{args.complexity_coeff}" if args.complexity_penality else "no_cp",
)

os.makedirs(base_dir, exist_ok=True)
log_path = os.path.join(base_dir, "log.txt")
with open(log_path, "a") as f:
    f.write("new run \n")

if args.task_name == "finetune":
    cmd = f"torchrun --nproc_per_node=1 src/gpt2_ft.py \
        --train_data ./data/e2e/train.jsonl \
        --valid_data ./data/e2e/valid.jsonl \
        --train_batch_size 8 \
        --grad_acc 1 \
        --valid_batch_size 4 \
        --seq_len 512 \
        --model_card gpt2.md \
        --init_checkpoint {args.init_checkpoint} \
        --platform local \
        --clip 0.0 \
        --lr 0.0002 \
        --weight_decay 0.01 \
        --correct_bias \
        --adam_beta2 0.999 \
        --scheduler linear \
        --warmup_step 500 \
        --max_epoch 5 \
        --save_interval 1000 \
        --lora_dim 4 \
        --lora_alpha 32 \
        --lora_dropout 0.1 \
        --label_smooth 0.1 \
        --work_dir {base_dir} \
        --random_seed 110 "
    if args.complexity_penality:
        cmd += " --use-complexity"
        cmd += f" --complexity-coeff {args.complexity_coeff}"
elif args.task_name == "retrain":
    cmd = f"torchrun --nproc_per_node=1 src/gpt2_prune_retrain.py \
        --train_data ./data/e2e/train.jsonl \
        --valid_data ./data/e2e/valid.jsonl \
        --train_batch_size 8 \
        --grad_acc 1 \
        --valid_batch_size 4 \
        --seq_len 512 \
        --model_card gpt2.md \
        --init_checkpoint {args.init_checkpoint} \
        --platform local \
        --clip 0.0 \
        --lr 0.0002 \
        --weight_decay 0.01 \
        --correct_bias \
        --adam_beta2 0.999 \
        --scheduler linear \
        --warmup_step 500 \
        --max_epoch 5 \
        --save_interval 1000 \
        --lora_dim 4 \
        --lora_alpha 32 \
        --lora_dropout 0.1 \
        --label_smooth 0.1 \
        --work_dir {base_dir} \
        --random_seed 110 \
        --lora_path {args.lora_path} \
        --n_pruning {args.n_pruning} \
        --percent_pruning {args.percent_pruning} \
        --thr_pruning {args.thr_pruning} "
elif args.task_name == "sparse":
    cmd = f"torchrun --nproc_per_node=1 src/gpt2_sparse_retrain.py \
        --train_data ./data/e2e/train.jsonl \
        --valid_data ./data/e2e/valid.jsonl \
        --train_batch_size 8 \
        --grad_acc 1 \
        --valid_batch_size 4 \
        --seq_len 512 \
        --model_card gpt2.md \
        --init_checkpoint {args.init_checkpoint} \
        --platform local \
        --clip 0.0 \
        --lr 0.0002 \
        --weight_decay 0.01 \
        --correct_bias \
        --adam_beta2 0.999 \
        --scheduler linear \
        --warmup_step 500 \
        --max_epoch 5 \
        --save_interval 1000 \
        --lora_dim 4 \
        --lora_alpha 32 \
        --lora_dropout 0.1 \
        --label_smooth 0.1 \
        --work_dir {base_dir} \
        --random_seed 110 "
else:
    raise NotImplementedError(f"Task: {args.task_name} is not supported.")

if args.enable_mlp_fc:
    cmd += " --enable_mlp_fc"
if args.enable_mlp_proj:
    cmd += " --enable_mlp_proj"
if args.enable_wo:
    cmd += " --enable_wo"
if args.enable_wq:
    cmd += " --enable_wq"
if args.enable_wk:
    cmd += " --enable_wk"
if args.enable_wv:
    cmd += " --enable_wv"

cmd += f"> {log_path}"
print(f"CMD: {cmd}")
subprocess.run(cmd, shell=True)
