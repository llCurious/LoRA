import torch
import torch.nn as nn
import transformers
from functools import partial
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from tqdm import tqdm
import functools
from collections import defaultdict
import numpy as np

from .data_utils import get_dataset

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
        
        if prefix == "transformer.h":
            scale_dict["attn_input_scale"] = act_dict[
                f"{prefix}.{idx}.attn.c_attn"]['input'] / 127
            scale_dict["attn_output_scale"] = act_dict[
                f"{prefix}.{idx}.attn.c_attn"]['output'] / 127
            scale_dict["out_input_scale"] = act_dict[
                f"{prefix}.{idx}.attn.c_proj"]['input'] / 127
            scale_dict["fc1_input_scale"] = act_dict[
                f"{prefix}.{idx}.mlp.c_fc"]['input'] / 127
            scale_dict["fc2_input_scale"] = act_dict[
                f"{prefix}.{idx}.mlp.c_proj"]["input"] / 127
        if prefix == "model.decoder.layers":
            scale_dict["attn_input_scale"] = act_dict[
                f"{prefix}.{idx}.self_attn.q_proj"]['input'] / 127
            scale_dict["q_output_scale"] = act_dict[
                f"{prefix}.{idx}.self_attn.q_proj"]['output'] / 127
            scale_dict["k_output_scale"] = act_dict[
                f"{prefix}.{idx}.self_attn.k_proj"]['output'] / 127
            scale_dict["v_output_scale"] = act_dict[
                f"{prefix}.{idx}.self_attn.v_proj"]['output'] / 127
            scale_dict["out_input_scale"] = act_dict[
                f"{prefix}.{idx}.self_attn.out_proj"]['input'] / 127
            scale_dict["fc1_input_scale"] = act_dict[
                f"{prefix}.{idx}.fc1"]['input'] / 127
            scale_dict["fc2_input_scale"] = act_dict[
                f"{prefix}.{idx}.fc2"]["input"] / 127
        
        decoder_layer_scales.append(scale_dict)

    return decoder_layer_scales, act_dict

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
                    partial(stat_input_hook, name=name))
            )

    dataset = get_dataset(dataset_name)
    dataset = dataset.shuffle(seed=42)

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
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear) or isinstance(fc, transformers.pytorch_utils.Conv1D)
        # assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    
    if isinstance(fc, transformers.pytorch_utils.Conv1D):
        # the weights shall be transposed
        weights = [fc.weight.t() for fc in fcs]
    elif isinstance(fc, nn.Linear):
        weights = [fc.weight for fc in fcs]

    weight_scales = torch.cat([weight.abs().max(
        dim=0, keepdim=True)[0] for weight in weights], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (act_scales.pow(alpha) / weight_scales.pow(1-alpha)
              ).clamp(min=1e-5).to(device).to(dtype)

    ln.weight.div_(scales)
    ln.bias.div_(scales)

    for fc, weight in zip(fcs, weights):
        if isinstance(fc, transformers.pytorch_utils.Conv1D):
            # the weights shall be transposed
            fc.weight.data = weight.mul_(scales.view(1, -1)).t()
        elif isinstance(fc, nn.Linear):
            fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5):
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [module.self_attn.q_proj,
                   module.self_attn.k_proj, module.self_attn.v_proj]
            qkv_input_scales = scales[name + '.self_attn.q_proj']
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + '.fc1']
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        elif isinstance(module, BloomBlock):
            attn_ln = module.input_layernorm
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + '.self_attention.query_key_value']
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm
            fc1 = module.mlp.dense_h_to_4h
            fc1_input_scales = scales[name + '.mlp.dense_h_to_4h']
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        elif isinstance(module, GPT2Block):
            attn_ln = module.ln_1
            qkv = module.attn.c_attn
            qkv_input_scales = scales[name + '.attn.c_attn']
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.ln_2
            fc1 = module.mlp.c_fc
            fc1_input_scales = scales[name + '.mlp.c_fc']
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
