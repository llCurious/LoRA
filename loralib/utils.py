#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np

from typing import Dict

from .layers import LoRALayer, PruneLayer


# Prune lora functionalities @zixiu
def lora_complexity(model: nn.Module):
    block_complexity, global_complexity = 0, 0
    for m in model.modules():
        if isinstance(m, PruneLayer):
            block_complexity += m.complexity()
    
    return block_complexity

def prune_lora(
    model: nn.Module,
    num_prune: int = None,
    percent_prune: float = None,
    thr_prune: float = None,
) -> None:
    loras, lora_layers, keep_flags = [], [], []
    for m in model.modules():
        if isinstance(m, PruneLayer) and hasattr(m, "lora_scaling"):
            loras.append(m.lora_scaling)
            lora_layers.append(m)
            keep_flags.append(True)

    if num_prune is not None:
        to_rm = np.argsort(loras)[:num_prune]
    elif percent_prune is not None:
        to_rm = np.argsort(loras)[: int(len(loras) * percent_prune)]
    elif thr_prune is not None:
        to_rm = [i for i, lora in enumerate(loras) if lora < thr_prune]

    for rm_idx in to_rm:
        keep_flags[rm_idx] = False

    for layer, keep_flag in zip(lora_layers, keep_flags):
        layer.set_keep_flag(keep_flag)


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = "none") -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == "none":
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k}
    elif bias == "all":
        return {
            k: my_state_dict[k] for k in my_state_dict if "lora_" in k or "bias" in k
        }
    elif bias == "lora_only":
        to_return = {}
        for k in my_state_dict:
            if "lora_" in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError
