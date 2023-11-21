#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import math
from typing import Optional, List
from abc import ABC, abstractmethod

def get_mask(data, method="neg", k=128):
    # print(f"Input data of shape: {data.shape}")

    if method == "neg":
        mask = data < 0
        return mask
    elif method == "topk":
        assert k is not None
        topk, indices = torch.topk(data, k, dim=-1)
        topk_mat = torch.zeros_like(data).scatter(-1, indices, topk)
        mask = topk_mat == 0
        return mask
    else:
        raise NotImplementedError(f"Get mask method: {method} is not supported.")
    
class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0,
            merge_weights=merge_weights,
        )
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(
                        0, 1
                    ) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(
                        0, 1
                    ) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            after_A = F.embedding(
                x,
                self.lora_A.transpose(0, 1),
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            # self.scaling = self.lora_alpha / self.r
            self.lora_scaling = nn.Parameter(torch.tensor(self.lora_alpha / self.r))

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.lora_scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.lora_scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (
                self.lora_dropout(x)
                @ self.lora_A.transpose(0, 1)
                @ self.lora_B.transpose(0, 1)
            ) * self.lora_scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )
        assert (
            out_features % len(enable_lora) == 0
        ), "The length of enable_lora must divide out_features"
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros(
                    (out_features // len(enable_lora) * sum(enable_lora), r)
                )
            )  # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features,), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0),
            self.lora_B.unsqueeze(-1),
            groups=sum(self.enable_lora),
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
            return result


class ConvLoRA(nn.Module, LoRALayer):
    def __init__(
        self,
        conv_module,
        in_channels,
        out_channels,
        kernel_size,
        r=0,
        lora_alpha=1,
        lora_dropout=0.0,
        merge_weights=True,
        **kwargs,
    ):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.conv.weight.new_zeros(
                    (out_channels // self.conv.groups * kernel_size, r * kernel_size)
                )
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(
                        self.conv.weight.shape
                    ) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(
                        self.conv.weight.shape
                    ) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x,
                self.conv.weight
                + (self.lora_B @ self.lora_A).view(self.conv.weight.shape)
                * self.scaling,
                self.conv.bias,
            )
        return self.conv(x)


class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)


class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)


# Can Extend to other ones like this


class Conv3d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)


# Custom Conv1D functions for GPT models
# class GPTConv1D(nn.Module):
#     def __init__(self, nf, nx):
#         super(GPTConv1D, self).__init__()
#         self.nf = nf
#         w = torch.empty(nx, nf)
#         nn.init.normal_(w, std=0.02)
#         self.weight = Parameter(w)
#         self.bias = Parameter(torch.zeros(nf))

#     def forward(self, x):
#         size_out = x.size()[:-1] + (self.nf,)
#         x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
#         x = x.view(*size_out)
#         return x


# Custom Conv1D functions for GPT models


class GPTConv1D(nn.Module, LoRALayer):
    # LoRA implemented in a Conv1D layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        sparsify_activation: bool = False,
        seq_length: int = 0,
        static_sparsity: bool = True,
        **kwargs,
    ):
        super(GPTConv1D, self).__init__()
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.in_features = in_features
        self.out_features = out_features
        self.fan_in_fan_out = fan_in_fan_out
        w = torch.empty(in_features, out_features)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(out_features))

        # extra sparsity parameters
        self.sparsify_activation = sparsify_activation
        self.seq_length = seq_length
        self.static_sparsity = static_sparsity

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            # self.lora_scaling = self.lora_alpha / self.r
            self.lora_scaling = nn.Parameter(torch.tensor(self.lora_alpha / self.r))

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        
        if self.sparsify_activation:
            self.mask_act = Parameter(torch.ones(self.seq_length, self.out_features).bool(), requires_grad=False)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.lora_scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.lora_scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        size_out = x.size()[:-1] + (self.out_features,)
        result = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        result = result.view(*size_out)

        tmp = result

        if self.r > 0 and not self.merged:
            result += (
                self.lora_dropout(x)
                @ self.lora_A.transpose(0, 1)
                @ self.lora_B.transpose(0, 1)
            ) * self.lora_scaling
        
        # sprsify the activation according to mask
        if self.sparsify_activation:
            if not self.static_sparsity:
                # print(f"Sparsify the activations with shape: {result.shape}")
                
                def get_common_position(data):
                    mask_list = []
                    n_samples = data.shape[0]
                    for i in range(n_samples):
                        # print(data[i])
                        mask_ = get_mask(data[i], method="neg")
                        # print(mask_)
                        mask_list.append(mask_)
                    return mask_list
                
                # 1. calculate mask
                mask_list = get_common_position(result)
                assert len(mask_list) > 0, f"Not enough sampels: n_sample: {len(mask_list)}"

                mask = self.mask_act.data
                for i in range(1, len(mask_list)):
                    mask = mask & mask_list[i]

                self.mask_act = Parameter(mask, requires_grad=False)
                print(f"Sparsify ratio: {torch.sum(self.mask_act).float() / self.mask_act.shape[0] / self.mask_act.shape[1]}")
            else:
            # 2. mask activation
                n_samples = result.shape[0]
                for i in range(n_samples):
                    # print(result[i])
                    result[i].data[self.mask_act] = 0
                    # print(result[i])
        return result, tmp


class PruneLayer:
    def __init__(self, keep_flag: bool = True):
        self.keep_flag = keep_flag

    def set_keep_flag(self, keep_flag: bool = True):
        self.keep_flag = keep_flag

    def get_keep_flag(self):
        return self.keep_flag

    @abstractmethod
    def complexity(self):
        pass

    @abstractmethod
    def empirical_consumption(self, hardwares):
        pass


class PruneMergedLinear(PruneLayer, MergedLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        keep_flag: List[bool] = [True],
        **kwargs,
    ):
        PruneLayer.__init__(self, keep_flag)
        # set enable_lora to True if both enable_lora and keep_flag is True
        enable_lora = [
            (enable_lora[i] and self.keep_flag[i]) for i in range(len(enable_lora))
        ]
        MergedLinear.__init__(
            self,
            in_features,
            out_features,
            r,
            lora_alpha,
            lora_dropout,
            enable_lora,
            fan_in_fan_out,
            merge_weights,
            **kwargs,
        )

        self.enable_lora = enable_lora
        self.scaling = nn.Parameter(torch.tensor(self.lora_alpha / self.r))

    def forward(self, x: torch.Tensor):
        return MergedLinear.forward(self, x)


class PruneGPTConv1D(PruneLayer, GPTConv1D):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        keep_flag: bool = True,
        name: str = None,
        sparsify_activation: bool = False,
        seq_length: int = 0,
        static_sparsity: bool = True,
        **kwargs,
    ):
        PruneLayer.__init__(self, keep_flag)
        GPTConv1D.__init__(
            self,
            in_features,
            out_features,
            r,
            lora_alpha,
            lora_dropout,
            fan_in_fan_out,
            merge_weights,
            sparsify_activation,
            seq_length,
            static_sparsity,
            **kwargs,
        )

        self.name = name
        # update scaling as thr
        # self.scaling = nn.Parameter(torch.tensor(self.lora_alpha / self.r))

    def forward(self, x: torch.Tensor, idx=None):
        if not self.keep_flag:
            self.merged = True  # set merged to True to escape computing lora module

        res, tmp = GPTConv1D.forward(self, x)
        # if idx is not None:
        #     print(f"{idx}-{self.name}-act.pt")
        #     torch.save(tmp, f"dist/{idx}-{self.name}-act.pt")
        return res

    def complexity(self):
        return self.lora_scaling * (
            self.r * self.in_features + self.out_features * self.r
        )

    def empirical_consumption(self, hardwares):
        return self.complexity()


class PruneLinear(PruneLayer, Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        keep_flag: bool = True,
        **kwargs,
    ):
        PruneLayer.__init__(self, keep_flag)
        Linear.__init__(
            self,
            in_features,
            out_features,
            r,
            lora_alpha,
            lora_dropout,
            fan_in_fan_out,
            merge_weights,
            **kwargs,
        )

        # update scaling as thr
        # self.scaling = nn.Parameter(torch.tensor(self.lora_alpha / self.r))

    def forward(self, x: torch.Tensor):
        if not self.keep_flag:
            self.merged = True  # set merged to True to escape computing lora module
        return Linear.forward(self, x)

    def complexity(self):
        return self.lora_scaling * (
            self.r * self.in_features + self.out_features * self.r
        )

    def empirical_consumption(self, hardwares):
        return self.complexity()
