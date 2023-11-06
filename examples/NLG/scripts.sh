set -x

enable_mlp=$1
enable_wo=$2
enable_wq=$3
enable_wk=$4
enable_wv=$5
complexity_penality=$6
complexity_coeff=0.0000001

cmd_suffix=" "
if [ "$enable_mlp" = true ]; then
    cmd_suffix=$cmd_suffix"--enable_mlp "
fi
if [ "$enable_wo" = true ]; then
    cmd_suffix=$cmd_suffix"--enable_wo "
fi
if [ "$enable_wq" = true ]; then
    cmd_suffix=$cmd_suffix"--enable_wq "
fi
if [ "$enable_wk" = true ]; then
    cmd_suffix=$cmd_suffix"--enable_wk "
fi
if [ "$enable_wv" = true ]; then
    cmd_suffix=$cmd_suffix"--enable_wv "
fi

if [ "$complexity_penality" = true ]; then
    cmd_suffix=$cmd_suffix"--use-complexity --complexity-coeff "$complexity_coeff
fi

echo $cmd_suffix

torchrun --nproc_per_node=1 src/gpt2_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --train_batch_size 8 \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
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
    --work_dir ./trained_models/GPT2_M/e2e \
    --random_seed 110 \
    $cmd_suffix \
    > lora_"$enable_mlp"_"$enable_wo"_"$enable_wq"_"$enable_wk"_"$enable_wv"_"$complexity_penality"_complexity_coeff_"$complexity_coeff".out
