import torch.nn as nn
from transformers.adapters.modeling import Activation_Function_Class
import torch
import torch.nn.functional as F
import os
from copy import deepcopy



is_amp_available = True
from torch.cuda.amp import autocast



class PrefixTuningConfig:
    prefix_length: int = 4
    bottleneck_size: int = 1024
    non_linearity: str = "relu"
    dropout: float = 0.0
    n_layers: int = 24
    n_heads: int = 16
    input_size: int = 1024


class PrefixTuning(nn.Module):
    def __init__(self, config: PrefixTuningConfig):
        super().__init__()
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.input_size = config.input_size
        self.n_embd_per_head = self.input_size // self.n_heads
        self.config = config

        self.input_tokens = torch.arange(self.config.prefix_length).long()
        self.wte = nn.Embedding(self.config.prefix_length, self.input_size)
        self.control_trans = nn.Sequential(
            nn.Linear(self.input_size, self.config.bottleneck_size),
            Activation_Function_Class(self.config.non_linearity.lower()),
            nn.Linear(self.config.bottleneck_size, self.n_layers * 2 * self.input_size),
        )

    def simple_forward(self):
        device = next(self.parameters()).device
        input_tokens = self.input_tokens.unsqueeze(0).expand(1, -1).to(device)
        embs = self.wte(input_tokens)
        key_values = self.control_trans(embs)  # 1 x prefix_length x n_layers*2*input_size
        key_values = key_values.view(
            1, self.config.prefix_length, self.n_layers * 2, self.n_heads, self.n_embd_per_head
        )  # *2 for key and value
        key_values = key_values.permute(2, 0, 3, 1, 4)
        # (n_layers * 2) x 1 x n_heads x  prefix_length  x n_embd_per_head
        return key_values

    def forward(self, batch_size):
        device = next(self.parameters()).device
        input_tokens = self.input_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        embs = self.wte(input_tokens)
        key_values = self.control_trans(embs)  # batch_size x prefix_length x n_layers*2*input_size
        key_values = key_values.view(
            batch_size, self.config.prefix_length, self.n_layers * 2, self.n_heads, self.n_embd_per_head
        )  # *2 for key and value
        # batch_size x prefix_length x (n_layers * 2) x n_heads x n_embd_per_head
        key_values = key_values.permute(2, 0, 3, 1, 4).split(2)
        # (n_layers * 2) x batch_size x n_heads x  prefix_length  x n_embd_per_head

        return key_values


from transformers.models.gpt2.modeling_gpt2 import GPT2MLP, GPT2Attention


class AffineNet(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim), True)
        self.bias = nn.Parameter(torch.zeros(hidden_dim), True)

    def forward(self, input_tensor):
        return input_tensor * self.weight + self.bias


class AffineGPT2Attention(nn.Module):
    def __init__(self, init_module: GPT2Attention):
        super().__init__()
        max_positions = 1024
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = init_module.embed_dim   if hasattr(init_module, "embed_dim")  else None   
        self.num_heads = init_module.num_heads   if hasattr(init_module, "num_heads")  else None      
        self.head_dim = init_module.head_dim   if hasattr(init_module, "head_dim")  else None    
        self.split_size = init_module.split_size    if hasattr(init_module, "split_size")  else None 
 
        self.scale_attn_weights = deepcopy(init_module.scale_attn_weights)
        self.scale_attn_by_inverse_layer_idx = deepcopy(init_module.scale_attn_by_inverse_layer_idx)
        self.is_cross_attention = deepcopy(init_module.cross_attention) if   hasattr(init_module, "cross_attention")  else None
        self.bias = deepcopy(init_module.bias)  if  hasattr(init_module, "bias")  else None
        self.masked_bias = deepcopy(init_module.masked_bias)  if  hasattr(init_module, "masked_bias")  else None
        self.layer_idx = deepcopy(init_module.layer_idx)    if  hasattr(init_module, "layer_idx")  else None
        self.attn_dropout = deepcopy(init_module.attn_dropout) if  hasattr(init_module, "attn_dropout")  else None
        self.q_attn = deepcopy(init_module.q_attn)   if  hasattr(init_module, "q_attn")  else None
        self.c_attn = deepcopy(init_module.c_attn)  if  hasattr(init_module, "c_attn")  else None

        self.reorder_and_upcast_attn = deepcopy(init_module.reorder_and_upcast_attn)  if  hasattr(init_module, "reorder_and_upcast_attn")  else None
        self.c_proj = deepcopy(init_module.c_proj)  if  hasattr(init_module, "c_proj")  else None
        self.resid_dropout = deepcopy(init_module.resid_dropout)  if  hasattr(init_module, "resid_dropout")  else None

        self.affine_net = AffineNet(self.embed_dim)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (value.size(-1) ** 0.5)

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

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

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        if is_amp_available:
            with autocast(enabled=False):
                q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
                attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
                attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)
        else:
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
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

        attn_output = self.affine_net(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class AffineGPT2MLP(nn.Module):
    def __init__(self, init_module: GPT2MLP, is_fine_grained=False):
        super().__init__()
        self.c_fc = deepcopy(init_module.c_fc)
        self.c_proj = deepcopy(init_module.c_proj)
        self.act = deepcopy(init_module.act)
        self.dropout = deepcopy(init_module.dropout)

        in_shape, intermediate_shape = self.c_fc.weight.shape

        self.affine_net_after = AffineNet(in_shape)

        if is_fine_grained:
            self.affine_net_intermediate = AffineNet(intermediate_shape)
        else:
            self.affine_net_intermediate = None

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)

        if self.affine_net_intermediate is not None:
            hidden_states = self.affine_net_intermediate(hidden_states)

        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.affine_net_after(hidden_states)
        return hidden_states


def traverse_net(model: nn.Module):
    n_layer = len(model.transformer.h)
    layer_ids = [_ for _ in range(n_layer)]
    
    for i in layer_ids:
      for name, sub_module in model.transformer.h[i].named_modules():
          if isinstance(sub_module, GPT2MLP):
                model.transformer.h[i].add_module(name, AffineGPT2MLP(sub_module, False))
                del sub_module


def traverse_fm_net(model: nn.Module):
    n_layer = len(model.transformer.h)
    layer_ids = [_ for _ in range(n_layer)]

    for i in layer_ids:
      for name, sub_module in model.transformer.h[i].named_modules():    
        if isinstance(sub_module, GPT2Attention):
              model.transformer.h[i].add_module(name, AffineGPT2Attention(sub_module))
              del sub_module



def traverse_fg_net(model: nn.Module):
    n_layer = len(model.transformer.h)
    layer_ids = [_ for _ in range(n_layer)]

    for i in layer_ids:
      for name, sub_module in model.transformer.h[i].named_modules():  
        if isinstance(sub_module, GPT2MLP):
              model.transformer.h[i].add_module(name, AffineGPT2MLP(sub_module, False))
              del sub_module
        elif isinstance(sub_module, GPT2Attention):
              model.transformer.h[i].add_module(name, AffineGPT2Attention(sub_module))
              del sub_module



def traverse_fgg_net(model: nn.Module):
    n_layer = len(model.transformer.h)
    layer_ids = [_ for _ in range(n_layer)]

    for i in layer_ids:
      for name, sub_module in model.transformer.h[i].named_modules(): 
        if isinstance(sub_module, GPT2MLP):
              model.transformer.h[i].add_module(name, AffineGPT2MLP(sub_module, True))
              del sub_module
        elif isinstance(sub_module, GPT2Attention):
              model.transformer.h[i].add_module(name, AffineGPT2Attention(sub_module))
              del sub_module



def traverse_lora_net(model: nn.Module):
    li = []
    for name, sub_module in model.named_children():
        if isinstance(sub_module, GPT2Attention):
            model.add_module(sub_module.c_attn._name_, LoRAConv1D(sub_module.c_attn))
        else:
            li.append(sub_module)
    for ele in li:
        traverse_lora_net(ele)


from transformers.modeling_utils import Conv1D
class LoRAConv1D(nn.Module):
    def __init__(self, init_module:Conv1D, intermediate_dim=6, scale=4):
        super().__init__()

        in_shape = Conv1D.weight.shape(0)

        self.linear = deepcopy(init_module)
        self.down_li = nn.ModuleList([nn.Linear(in_shape, intermediate_dim, bias=False) for _ in range(3)])
        self.up_li = nn.ModuleList([nn.Linear(intermediate_dim, in_shape, bias=False) for _ in range(3)])

        for down_net in self.down_li:
            down_net.weight = torch.randn([self.linear.in_features, intermediate_dim])

        for up_net in self.up_li:
            up_net.weight = torch.zeros([intermediate_dim, self.linear.in_features])

        self.scale = scale

    def forward(self, input_tensor):
        lora = []
        for down_net, up_net in zip(self.down_li, self.up_li):
            lora.append(up_net(down_net(input_tensor)))

        #bs, seq_len, dim
        lora = torch.stack(lora, dim=2)

        return self.linear(input_tensor) + self.scale * lora


