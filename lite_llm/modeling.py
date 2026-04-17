"""Small Llama-style causal LM with Multi-head Latent Attention.

Design choices:
- RMSNorm done entirely in fp32, cast back at the very end (Llama style).
- RoPE cos/sin buffers built once at ``__init__`` for ``max_position_embeddings``
  and never rebuilt at runtime.
- MLA: Q / K / V are produced from low-rank latent projections before being
  expanded into per-head tensors.
- Weight init: N(0, initializer_range). Residual projections (``o_proj``,
  ``down_proj``) are scaled by 1/sqrt(2 * num_hidden_layers) (GPT-2 style).
- KV cache: forward accepts ``past_key_values`` as a ``transformers.Cache``
  object for fast incremental decoding with ``model.generate()``.
"""

from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration import LiteLlmConfig


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(orig_dtype)


# ---------------------------------------------------------------------------
# Rotary Position Embedding (Llama-style half-split)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]
        return cos.to(x.dtype), sin.to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k: [B, H, S, D]; cos, sin: [S, D]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Multi-head Latent Attention
# ---------------------------------------------------------------------------

class MLAAttention(nn.Module):
    def __init__(self, config: LiteLlmConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank

        self.q_down_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_up_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.kv_down_proj = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=False)
        self.k_up_proj = nn.Linear(self.kv_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.v_up_proj = nn.Linear(self.kv_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.qk_norm = config.qk_norm
        if self.qk_norm:
            # QK-Norm (stability trick from Chameleon / Qwen2)
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape

        q_latent = self.q_down_proj(hidden_states)
        kv_latent = self.kv_down_proj(hidden_states)

        q = self.q_up_proj(q_latent).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_up_proj(kv_latent).view(bsz, q_len, self.num_heads, self.head_dim)
        v = self.v_up_proj(kv_latent).view(bsz, q_len, self.num_heads, self.head_dim)

        # [B, H, S, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cos, sin = self.rotary_emb(q, position_ids)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        # SDPA handles causal + additive mask. When no custom mask is given and
        # we are doing a fresh forward (no cache), use is_causal for the faster
        # kernel path.
        kv_len = k.shape[-2]
        is_causal = attention_mask is None and q_len > 1 and q_len == kv_len

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output)


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    def __init__(self, config: LiteLlmConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, config: LiteLlmConfig, layer_idx: int):
        super().__init__()
        self.attention = MLAAttention(config, layer_idx=layer_idx)
        self.ffn = SwiGLUFFN(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.attention(
            h,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        hidden_states = residual + h

        residual = hidden_states
        h = self.post_attention_layernorm(hidden_states)
        h = self.ffn(h)
        hidden_states = residual + h
        return hidden_states


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class LiteLlmModel(nn.Module):
    def __init__(self, config: LiteLlmConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape

        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + seq_len,
                device=input_ids.device, dtype=torch.long,
            )

        sdpa_mask = self._build_sdpa_mask(
            attention_mask, bsz, seq_len, past_len, input_ids.device,
        )

        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    partial(
                        layer.__call__,
                        attention_mask=sdpa_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                    ),
                    hidden_states,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=sdpa_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                )

        hidden_states = self.norm(hidden_states)
        return hidden_states

    @staticmethod
    def _build_sdpa_mask(attention_mask, bsz, q_len, past_len, device):
        """Convert an HF-style [B, S_full] 0/1 padding mask into a bool mask for
        SDPA. Returns None when nothing to mask beyond plain causal, so SDPA can
        take the ``is_causal=True`` fast path.
        """
        if attention_mask is None:
            return None
        kv_len = past_len + q_len
        mask = attention_mask.bool()
        if mask.dim() == 2:
            mask = mask[:, None, None, :].expand(bsz, 1, q_len, kv_len)
        causal = torch.ones(q_len, kv_len, device=device, dtype=torch.bool).tril(
            diagonal=past_len
        )
        return mask & causal


# ---------------------------------------------------------------------------
# Causal LM head
# ---------------------------------------------------------------------------

class LiteLlmForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = LiteLlmConfig
    base_model_prefix = "model"
    _no_split_modules = ["TransformerBlock"]
    supports_gradient_checkpointing = True
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: LiteLlmConfig):
        super().__init__(config)
        self.model = LiteLlmModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    # --- HF init hook ---
    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def post_init(self):
        super().post_init()
        # Scale residual projections by 1/sqrt(2 * num_hidden_layers) (GPT-2 style).
        scale = (2.0 * self.config.num_hidden_layers) ** -0.5
        for layer in self.model.layers:
            layer.attention.o_proj.weight.data.mul_(scale)
            layer.ffn.down_proj.weight.data.mul_(scale)

    # --- weight tying ---
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # --- forward ---
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if use_cache is None:
            use_cache = self.config.use_cache and not self.training

        # Training ignores the KV cache to keep memory/compute predictable.
        if not use_cache:
            past_key_values = None

        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
        )

    # --- generation support ---
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            # Only the last token needs to be fed when the cache already
            # contains the prefix.
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
        }
