import torch
from torch import nn
import math

from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Config,
    GPT2PreTrainedModel,
    GPT2Attention,
    GPT2Block,
    GPT2Model,
    GPT2MLP,
    GPT2LMHeadModel,
    GPT2DoubleHeadsModel,
    GPT2ForSequenceClassification,
    GPT2ForTokenClassification,
    GPT2ForQuestionAnswering,
)
from transformers.pytorch_utils import Conv1D
from typing import Optional, Tuple, List, Union
# import sys
# sys.path.append("/home/haoqi.whq/llm-inference/torch-int")
from torch_int.nn.linear import W8A8B8O8Conv1D, W8A8BFP32OFP32Conv1D, W8A8B8O8Conv1DReLU
from torch_int.nn.fused import LayerNormQ
from transformers.utils import logging
from torch_int.nn.bmm import BMM_S8T_S8N_S8T, BMM_S8T_S8N_F32T
logger = logging.get_logger(__name__)

def gelu(x):
    return (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )


def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def gelu_new(x):
    """Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
    Also see https://arxiv.org/abs/1606.08415
    """
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )


def swish(x):
    return x * torch.sigmoid(x)


def _gelu_python(x):
    """Original Implementation of the gelu activation function in Google Bert repo when initially created.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    This is now written in C in torch.nn.functional
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Int8GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        # if self.is_cross_attention:
        #     self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
        #     self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        # else:
        #     self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)

        # self.q_attn = W8A8B8O8Linear(self.embed_dim, self.embed_dim)
        # self.k_attn = W8A8B8O8Linear(self.embed_dim, self.embed_dim)
        # self.v_attn = W8A8B8O8Linear(self.embed_dim, self.embed_dim)
        self.c_attn = W8A8B8O8Conv1D(3 * self.embed_dim, self.embed_dim)

        self.qk_bmm = BMM_S8T_S8N_F32T(1.0)
        self.pv_bmm = BMM_S8T_S8N_S8T(1.0)

        self.c_proj = W8A8BFP32OFP32Conv1D(self.embed_dim, self.embed_dim)
        # self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    @staticmethod
    def from_float(module,
                   config,
                   input_scale: float,
                   output_scale: float,
                   out_input_scale: float):
        
        int8_module = Int8GPT2Attention(config, is_cross_attention=False, layer_idx=None)
        # Fuse the scaling into the q_proj output scale
        # q_output_scale = q_output_scale * module.scaling
        # module.q_attn.weight *= module.scaling
        # module.q_attn.bias *= module.scaling
        int8_module.c_attn = W8A8B8O8Conv1D.from_float(
            module.c_attn, 3 * config.hidden_size, config.hidden_size, input_scale, output_scale)
        # int8_module.k_attn = W8A8B8O8Linear.from_float(
        #     module.k_attn, input_scale, k_output_scale)
        # int8_module.v_attn = W8A8B8O8Linear.from_float(
        #     module.v_attn, input_scale, v_output_scale)
        
        int8_module.qk_bmm = BMM_S8T_S8N_F32T.from_scale(
            output_scale, output_scale)
        # alpha = s_prob * s_v / s_out, where s_prob = 1 / 127
        int8_module.pv_bmm = BMM_S8T_S8N_S8T.from_scale(
            1.0 / 127, output_scale, out_input_scale)
        
        int8_module.c_proj = W8A8BFP32OFP32Conv1D.from_float(
            module.c_proj, config.hidden_size, config.hidden_size, out_input_scale)

        return int8_module

    prune_heads = GPT2Attention.prune_heads
    _attn = GPT2Attention._attn
    _upcast_and_reordered_attn = GPT2Attention._upcast_and_reordered_attn
    _split_heads = GPT2Attention._split_heads
    _merge_heads = GPT2Attention._merge_heads
    

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        bsz = query.size()[0]
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)

        query = query.view(*proj_shape)
        key = key.view(*proj_shape)
        value = value.view(*proj_shape)
        attn_weights = self.qk_bmm(query, key)

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # (A_row V_row)_row = (A_row V_col ^T)_row
        attn_weights.mul_(127).round_()
        attn_weights = attn_weights.to(torch.int8)

        value = value.transpose(1, 2).contiguous()
        attn_weights = attn_weights.view(bsz*self.num_heads, attn_weights.size()[-2], attn_weights.size()[-1])
        attn_output = self.pv_bmm(attn_weights, value)
        attn_output = attn_output.view(
            bsz, self.num_heads, -1, self.head_dim)
        # attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)
    
class Int8GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        hidden_size = config.hidden_size
        # self.c_fc = Conv1D(intermediate_size, embed_dim)
        # self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.c_fc = W8A8B8O8Conv1DReLU(intermediate_size, hidden_size)
        self.c_proj = W8A8BFP32OFP32Conv1D(hidden_size, intermediate_size)
        # self.act = ACT2FN[config.activation_function]
        self.act = torch.nn.ReLU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    @staticmethod
    def from_float(module: GPT2MLP, config, fc1_input_scale, fc2_input_scale):
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        
        int8_module = Int8GPT2MLP(inner_dim, config)

        int8_module.c_fc = W8A8B8O8Conv1DReLU.from_float(
            module.c_fc, inner_dim, hidden_size, fc1_input_scale, fc2_input_scale)
        int8_module.c_proj = W8A8BFP32OFP32Conv1D.from_float(
            module.c_proj, hidden_size,inner_dim, fc2_input_scale)
        return int8_module

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        # hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
    
class Int8GPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.attn = Int8GPT2Attention(config, layer_idx=layer_idx)

        self.ln_1 = LayerNormQ(hidden_size, eps=config.layer_norm_epsilon)
        self.ln_2 = LayerNormQ(hidden_size, eps=config.layer_norm_epsilon)
        
        self.mlp = Int8GPT2MLP(inner_dim, config)

    @staticmethod
    def from_float(module: GPT2Block,
                   config: GPT2Config,
                   attn_input_scale: float,
                   attn_output_scale: float,
                   out_input_scale: float,
                   fc1_input_scale: float,
                   fc2_input_scale: float):
        int8_module = Int8GPT2Block(
            config
        )
        int8_module.ln_1 = LayerNormQ.from_float(
            module.ln_1, attn_input_scale)
        int8_module.attn = Int8GPT2Attention.from_float(
            module.attn, config, attn_input_scale, attn_output_scale, out_input_scale)
        int8_module.ln_2 = LayerNormQ.from_float(
            module.ln_2, fc1_input_scale)
        int8_module.mlp = Int8GPT2MLP.from_float(module.mlp, config, fc1_input_scale, fc2_input_scale)
        return int8_module

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)
    
class Int8GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Int8GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
    get_input_embeddings = GPT2Model.get_input_embeddings
    set_input_embeddings = GPT2Model.set_input_embeddings
    parallelize = GPT2Model.parallelize
    deparallelize = GPT2Model.deparallelize
    _prune_heads = GPT2Model._prune_heads
    forward = GPT2Model.forward

    @staticmethod
    def from_float(module: GPT2Model, decoder_layer_scales):
        int8_module = Int8GPT2Model(module.config)
        int8_module.wte = module.wte
        int8_module.wpe = module.wpe
        int8_module.drop = module.drop
        int8_module.ln_f = module.ln_f
        for i, layer in enumerate(module.h):
            int8_module.h[i] = Int8GPT2Block.from_float(
                layer, module.config, **decoder_layer_scales[i])
        return int8_module
    
class Int8GPT2LMHeadModel(GPT2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = Int8GPT2Model(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def from_float(module: GPT2LMHeadModel, decoder_layer_scales):
        int8_module = Int8GPT2LMHeadModel(module.config)
        int8_module.transformer = Int8GPT2Model.from_float(
            module.transformer, decoder_layer_scales)
        int8_module.lm_head = module.lm_head
        return int8_module

    get_output_embeddings = GPT2LMHeadModel.get_output_embeddings
    set_output_embeddings = GPT2LMHeadModel.set_output_embeddings

    parallelize = GPT2LMHeadModel.parallelize
    deparallelize = GPT2LMHeadModel.deparallelize

    prepare_inputs_for_generation = GPT2LMHeadModel.prepare_inputs_for_generation
    
    forward = GPT2LMHeadModel.forward
    _reorder_cache = GPT2LMHeadModel._reorder_cache