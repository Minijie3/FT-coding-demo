from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
import torch
import transformers

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.models.llama.modeling_llama import *

from prismatic.vla.constants import ACTION_CHUNK_LENGTH, ACTION_DIM

def llama_spda_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    num_vision: int = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    '''
    Replace the attention mask in the forward process of SPDA in Llama with a mask matrix that meets the requirements of CAttn.
    For the vision - language part, retain the causal mask. For the action part, remove the causal mask restriction.
    '''
    # `attention_mask` cannot be None (modified later)
    if attention_mask is None:
        attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        attention_mask = _prepare_4d_causal_attention_mask(attention_mask, hidden_states.shape[:2], hidden_states, past_key_value_length=0) # (bts, num_heads, q_len, q_len)

    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    batch_size, q_len, _ = hidden_states.size()

    # Grouped-Query Attention for key and value: each key-value header will be shared by multiple query headers
    # Number of queries: num_heads
    # Number of keys/values: num_key_values_groups
    query_states = self.q_proj(hidden_states) # (bts, q_len, num_heads * head_dim)
    key_states = self.k_proj(hidden_states) # (bts, q_len, num_key_value_heads * head_dim), same same as `value_states`
    value_states = self.v_proj(hidden_states)
    query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2) # (bts, num_heads, q_len, head_dim)
    key_states = key_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # (bsz, num_key_value_heads, q_len, head_dim), same as `value_states`
    value_states = value_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    # shape of `position_idx`: (bts, q_len)
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states)

    past_key_values = getattr(self, "past_key_values", past_key_values)
    if past_key_values is not None:
        cache_kwargs = {"sin":sin, "cos": cos, "cache_position": cache_position}
        '''
        process: 
            past_key = torch.cat([past_key, key_states], dim=2)  # dim=2 ==> q_len
            past_value = torch.cat([past_value, value_states], dim=2)
        get key/value: (bsz, num_key_value_heads, total_seq_len, head_dim) while total_seq_len = past_len + current_seq_len
        '''
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    
    # Align for GQA: num_key_value_groups * num_key_value_heads = num_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups) 

    causal_mask = attention_mask
    if causal_mask is not None:
        causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    if causal_mask is not None:
        min_dtype = roch.finfo(hidden_states.dtype).min # `-infty` for ignoration
        num_pad = (causal_mask[:, 0, -1, :] == min_dtype).sum(dim = 1) # (batch_size,)
        num_act = ACTION_CHUNK_LENGTH * ACTION_DIM + 1 # `+1` for stop token
        
        mask = causal_mask.clone()
        for idx, n_pad in enumerate(num_pad):
            '''
            Causal Mask Example:
                No pad tokens:
                    0 -inf -inf
                    0   0  -inf
                    0   0    0
                1 pad token:
                    0 -inf -inf
                    0   0  -inf
                    0   0  -inf
                2 pad tokens:
                    0 -inf -inf
                    0 -inf -inf
                    0 -inf -inf
            So start = -(num_act + n_pad), end = -n_pad when n_pad != 0
            '''
            mask[idx, :, -num_act:, -num_act:] = 0 if n_pad == 0 else (
                mask[idx, :, -(num_act + n_pad):-n_pad, -(num_act + n_pad):-n_pad,] = 0
            )
        causal_mask = mask

    output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=False,
    ).transpose(1, 2).contiguous().view(batch_size, q_len, self.hidden_size)
    output = self.o_proj(output)

    return output, None, past_key_values

def replace_llama_spda_forward():
    transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attention_forward

