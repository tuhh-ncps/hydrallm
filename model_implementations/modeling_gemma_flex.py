# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# Reuse Gemma3 internals from transformers
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3PreTrainedModel,
    Gemma3TextConfig,
    Gemma3Config,
    Gemma3RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
    DynamicCache,
    Cache,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    GradientCheckpointingLayer,
    create_causal_mask,
    create_sliding_window_causal_mask,
    Gemma3ModelOutputWithPast,
    Gemma3CausalLMOutputWithPast,
)
from transformers.masking_utils import (
    create_masks_for_generate as hf_create_masks_for_generate,
)
from transformers.utils import logging
from transformers.generation import GenerationMixin
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers import AutoModel, PretrainedConfig


# Local import from same folder
try:
    from modeling_common import (
        DynamicLinear,
        DynamicRMSNorm,
        DynamicTextScaledWordEmbedding,
    )
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from modeling_common import (
        DynamicLinear,
        DynamicRMSNorm,
        DynamicTextScaledWordEmbedding,
    )


logger = logging.get_logger(__name__)


def _cache_is_initialized_safe(past_key_values) -> bool:
    """
    Transformers versions differ: some expose DynamicCache.is_initialized, some don't.
    Fall back to get_seq_length() > 0 if the attribute is missing.
    """
    if past_key_values is None:
        return False
    if hasattr(past_key_values, "is_initialized"):
        return bool(getattr(past_key_values, "is_initialized"))
    return getattr(past_key_values, "get_seq_length", lambda: 0)() > 0


class FlexCoreGemma3MLP(nn.Module):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = DynamicLinear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = DynamicLinear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = DynamicLinear(
            self.intermediate_size, self.hidden_size, bias=False
        )
        from transformers.activations import ACT2FN

        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x: torch.Tensor, dyn_embed_dim: int, dyn_intermediate_size: int):
        gate = self.gate_proj.forward_sliced(
            x, out_rows=dyn_intermediate_size, in_cols=dyn_embed_dim
        )
        up = self.up_proj.forward_sliced(
            x, out_rows=dyn_intermediate_size, in_cols=dyn_embed_dim
        )
        hidden = self.act_fn(gate) * up
        out = self.down_proj.forward_sliced(
            hidden, out_rows=dyn_embed_dim, in_cols=dyn_intermediate_size
        )
        return out


class FlexCoreGemma3Attention(nn.Module):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.is_causal = not getattr(config, "use_bidirectional_attention", False)
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )
        self.q_proj = DynamicLinear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = DynamicLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = DynamicLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = DynamicLinear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = DynamicRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = DynamicRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_logit_softcapping = self.config.attn_logit_softcapping

    # This might be faster with different attention, but there were some problems. TODO: Check this again! (SDPA | FlashAttention)
    def _attn_impl(self):
        impl = (
            getattr(self.config, "_attn_implementation", None)
            or getattr(self.config, "attn_implementation", None)
            or "eager"
        )
        if impl in (None, "eager"):
            return eager_attention_forward
        try:
            return ALL_ATTENTION_FUNCTIONS[impl]
        except KeyError:
            return eager_attention_forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        current_num_heads: int = None,
        dyn_embed_dim: int = None,
        **kwargs,
    ):
        # 1. Determine Dimensions
        num_heads = (
            max(1, min(current_num_heads, self.num_attention_heads))
            if current_num_heads
            else self.num_attention_heads
        )

        # Calculate full group size (e.g., 14 heads / 2 kv = 7)
        full_group_size = self.num_attention_heads // self.num_key_value_heads

        # Calculate required KV heads (e.g., 10 Q heads -> 2 KV heads)
        num_kv_heads_needed = math.ceil(num_heads / full_group_size)

        q_out_dim = num_heads * self.head_dim
        kv_out_dim = num_kv_heads_needed * self.head_dim
        input_shape = hidden_states.shape[:-1]

        # 2. Sliced Projections
        q = self.q_proj.forward_sliced(
            hidden_states, out_rows=q_out_dim, in_cols=dyn_embed_dim
        )
        k = self.k_proj.forward_sliced(
            hidden_states, out_rows=kv_out_dim, in_cols=dyn_embed_dim
        )
        v = self.v_proj.forward_sliced(
            hidden_states, out_rows=kv_out_dim, in_cols=dyn_embed_dim
        )

        # Reshape to [Batch, Heads, Seq, HeadDim]
        q = q.view(*input_shape, num_heads, self.head_dim).transpose(1, 2)
        k = k.view(*input_shape, num_kv_heads_needed, self.head_dim).transpose(1, 2)
        v = v.view(*input_shape, num_kv_heads_needed, self.head_dim).transpose(1, 2)

        # 3. RoPE & Cache
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # 4. Hybrid Group Handling
        # Calculate how many times each KV head must be repeated
        repeats = []
        heads_remaining = num_heads
        for _ in range(num_kv_heads_needed):
            count = min(heads_remaining, full_group_size)
            repeats.append(count)
            heads_remaining -= count

        # Check if all groups are the same size (Uniform GQA)
        is_uniform = all(r == repeats[0] for r in repeats)

        original_num_kv_groups = self.num_key_value_groups

        if is_uniform:
            # Path A: Uniform Groups (e.g., 4Q, 1KV or 14Q, 2KV)
            # Use implicit broadcasting (efficient & numerically matches baseline)
            self.num_key_value_groups = repeats[0]

            attn_output, attn_weights = self._attn_impl()(
                self,
                q,
                k,
                v,
                attention_mask,
                dropout=self.attention_dropout if self.training else 0.0,
                scaling=self.scaling,
                softcap=self.attn_logit_softcapping,
                sliding_window=self.sliding_window,
                **kwargs,
            )
        else:
            # Path B: Uneven Groups (e.g., 10Q, 2KV -> groups of 7 and 3)
            # Manual expansion required
            repeat_tensor = torch.tensor(repeats, device=k.device, dtype=torch.long)
            k_expanded = torch.repeat_interleave(k, repeat_tensor, dim=1)
            v_expanded = torch.repeat_interleave(v, repeat_tensor, dim=1)

            # Pretend we are MHA (1 group per head) since we manually expanded
            self.num_key_value_groups = 1

            attn_output, attn_weights = self._attn_impl()(
                self,
                q,
                k_expanded,
                v_expanded,
                attention_mask,
                dropout=self.attention_dropout if self.training else 0.0,
                scaling=self.scaling,
                softcap=self.attn_logit_softcapping,
                sliding_window=self.sliding_window,
                **kwargs,
            )

        # Restore original state
        self.num_key_value_groups = original_num_kv_groups

        # 5. Output Projection
        attn_output = attn_output.reshape(*input_shape, q_out_dim)
        out = self.o_proj.forward_sliced(
            attn_output, out_rows=dyn_embed_dim, in_cols=q_out_dim
        )

        return out, attn_weights


class FlexCoreGemma3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.input_layernorm = DynamicRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn = FlexCoreGemma3Attention(config, layer_idx)
        self.post_attention_layernorm = DynamicRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = DynamicRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = FlexCoreGemma3MLP(config)
        self.post_feedforward_layernorm = DynamicRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: Tuple[torch.Tensor, torch.Tensor],
        position_embeddings_local: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        current_num_heads: int = None,
        dyn_embed_dim: int = None,
        dyn_intermediate_size: int = None,
        **kwargs,
    ):
        # Fix: Remove position_embeddings from kwargs to avoid conflict with implicit pos_emb construction
        kwargs.pop("position_embeddings", None)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states, in_cols=dyn_embed_dim)
        pos_emb = (
            position_embeddings_local
            if self.attention_type == "sliding_attention"
            else position_embeddings_global
        )

        attn_out, attn_weights = self.self_attn(
            hidden_states,
            position_embeddings=pos_emb,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            current_num_heads=current_num_heads,
            dyn_embed_dim=dyn_embed_dim,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(attn_out, in_cols=dyn_embed_dim)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(
            hidden_states, in_cols=dyn_embed_dim
        )
        hidden_states = self.mlp(
            hidden_states,
            dyn_embed_dim=dyn_embed_dim,
            dyn_intermediate_size=dyn_intermediate_size,
        )
        hidden_states = self.post_feedforward_layernorm(
            hidden_states, in_cols=dyn_embed_dim
        )
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class FlexCoreGemma3TextModel(Gemma3PreTrainedModel):
    config: Gemma3TextConfig

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = DynamicTextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [
                FlexCoreGemma3DecoderLayer(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = DynamicRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config=config)
        import copy

        cfg_local = copy.deepcopy(config)
        cfg_local.rope_theta = cfg_local.rope_local_base_freq
        cfg_local.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=cfg_local)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        current_num_heads: int = None,
        dyn_embed_dim: int = None,
        dyn_intermediate_size: int = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Fix: Remove position_embeddings from kwargs to avoid conflict downstream
        kwargs.pop("position_embeddings", None)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
        if self.gradient_checkpointing and self.training and output_attentions:
            logger.warning_once(
                "`output_attentions=True` is incompatible with gradient checkpointing during training. Setting `output_attentions=False`."
            )
            output_attentions = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids, out_cols=dyn_embed_dim)
        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache(config=self.config)
        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = dict(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        if isinstance(dyn_intermediate_size, (list, tuple)):
            if len(dyn_intermediate_size) != len(self.layers):
                raise ValueError(
                    "dyn_intermediate_size must contain one value per decoder layer"
                )
            layer_intermediate_sizes = dyn_intermediate_size
        else:
            layer_intermediate_sizes = [dyn_intermediate_size] * len(self.layers)

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_intermediate_size = layer_intermediate_sizes[layer_idx]
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module, intermediate_size):
                    def custom_forward(hidden_states):
                        return module(
                            hidden_states,
                            position_embeddings_global=position_embeddings_global,
                            position_embeddings_local=position_embeddings_local,
                            attention_mask=causal_mask_mapping[
                                decoder_layer.attention_type
                            ],
                            position_ids=position_ids,
                            past_key_values=None,
                            output_attentions=False,
                            use_cache=False,
                            cache_position=cache_position,
                            current_num_heads=current_num_heads,
                            dyn_embed_dim=dyn_embed_dim,
                            dyn_intermediate_size=intermediate_size,
                            **kwargs,
                        )[0]

                    return custom_forward

                hidden_states = checkpoint(
                    create_custom_forward(decoder_layer, layer_intermediate_size),
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings_global=position_embeddings_global,
                    position_embeddings_local=position_embeddings_local,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    current_num_heads=current_num_heads,
                    dyn_embed_dim=dyn_embed_dim,
                    dyn_intermediate_size=layer_intermediate_size,
                    **kwargs,
                )
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states, in_cols=dyn_embed_dim)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class FlexCoreGemma3ForCausalLM(Gemma3PreTrainedModel, GenerationMixin):
    config: Gemma3TextConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.model = FlexCoreGemma3TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = DynamicLinear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        current_num_heads: int = None,
        dyn_embed_dim: int = None,
        dyn_intermediate_size: int = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            current_num_heads=current_num_heads,
            dyn_embed_dim=dyn_embed_dim,
            dyn_intermediate_size=dyn_intermediate_size,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head.forward_sliced(hidden_states, in_cols=dyn_embed_dim)

        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits) * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        return super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, **kwargs
        )


# ----------------------------
# Multimodal extensions below
# ----------------------------


class FlexCoreGemma3MultiModalProjector(nn.Module):
    """
    Dynamic multimodal projector that maps vision hidden -> text hidden with optional slicing.
    Closely follows the projector behavior in Gemma3 [1][2], but uses a plain nn.Parameter
    (matching the HF checkpoint format) with manual output-column slicing.
    """

    def __init__(self, config: Gemma3Config):
        super().__init__()
        # Match HF checkpoint parameter name and shape exactly:
        # HF stores a single parameter with name `mm_input_projection_weight` shaped [vision_hidden, text_hidden]
        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(
                config.vision_config.hidden_size, config.text_config.hidden_size
            )
        )
        # Soft embedding norm on vision hidden size
        self.mm_soft_emb_norm = DynamicRMSNorm(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )

        self.patches_per_image = int(
            config.vision_config.image_size // config.vision_config.patch_size
        )
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(
            kernel_size=self.kernel_size, stride=self.kernel_size
        )

    def forward(
        self,
        vision_outputs: torch.Tensor,
        out_cols: int,
        final_embed_dim: int,
    ):
        """
        Args:
            vision_outputs: (batch, num_patches, vision_hidden) from vision tower last_hidden_state (as in SigLIP).
            out_cols: target projection output width (slice of text hidden, i.e. number of output columns).
            final_embed_dim: the LM dynamic hidden width expected for image tokens. If out_cols != final_embed_dim,
                             we will pad/truncate to match final_embed_dim for safe scatter into the text stream.
        """
        batch_size, _, seq_length = vision_outputs.shape

        reshaped = vision_outputs.transpose(1, 2)
        reshaped = reshaped.reshape(
            batch_size, seq_length, self.patches_per_image, self.patches_per_image
        ).contiguous()

        pooled = self.avg_pool(reshaped)
        pooled = pooled.flatten(2).transpose(1, 2)  # (B, tokens, vision_hidden)

        # RMSNorm on vision hidden
        normed = self.mm_soft_emb_norm(
            # full vision hidden size
            pooled,
            in_cols=self.mm_soft_emb_norm.weight.size(0),
        )

        # Dynamic projection: select the first `out_cols` output columns and matmul like HF
        # normed: [B, tokens, vision_hidden], weight: [vision_hidden, text_hidden]
        # we slice the OUTPUT dimension (second dim of the weight)
        projected = torch.matmul(normed, self.mm_input_projection_weight[:, :out_cols])
        # Now pad/truncate to final_embed_dim if necessary
        if projected.size(-1) < final_embed_dim:
            pad = final_embed_dim - projected.size(-1)
            projected = torch.nn.functional.pad(projected, (0, pad))
        elif projected.size(-1) > final_embed_dim:
            projected = projected[..., :final_embed_dim]

        return projected.type_as(vision_outputs)


def token_type_ids_mask_function(
    token_type_ids: Optional[torch.Tensor],
    image_group_ids: Optional[torch.Tensor],
    tokens_per_image: int,
):
    """
    Same semantics as in HF Gemma3: add an additional 'or' mask function to allow
    bidirectional attention within the same image block [1][2].
    """
    if token_type_ids is None:
        return None

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        safe_idx = torch.where(kv_idx < token_type_ids.shape[1], kv_idx, 0)
        token_type_ids_at_kv_idx = token_type_ids[batch_idx, safe_idx]
        token_type_ids_at_kv_idx = torch.where(
            kv_idx < token_type_ids.shape[1], token_type_ids_at_kv_idx, 0
        )

        image_group_ids_at_kv_idx = image_group_ids[batch_idx, safe_idx]
        image_group_ids_at_kv_idx = torch.where(
            kv_idx < image_group_ids.shape[1], image_group_ids_at_kv_idx, -1
        )

        is_image_block = (token_type_ids[batch_idx, q_idx] == 1) & (
            token_type_ids_at_kv_idx == 1
        )
        same_image_block = (
            image_group_ids[batch_idx, q_idx] == image_group_ids_at_kv_idx
        )
        return is_image_block & same_image_block

    return inner_mask


class FlexCoreGemma3Model(Gemma3PreTrainedModel):
    """
    Multimodal core model: vision tower + dynamic projector + dynamic text model.
    Mirrors Gemma3Model logic (token-type masks, merges image features, etc.) [1][2].
    """

    # We are filtering the logits/labels so we shouldn't divide the loss based on num_items_in_batch
    accepts_loss_kwargs = False

    def __init__(self, config: Gemma3Config):
        super().__init__(config)
        # Keep the vision encoder as AutoModel
        self.vision_tower = AutoModel.from_config(config=config.vision_config)
        self.multi_modal_projector = FlexCoreGemma3MultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size

        # Use our dynamic text model
        self.language_model = FlexCoreGemma3TextModel(config.text_config)

        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_image_features(
        self, pixel_values: torch.Tensor, proj_out_cols: int, final_embed_dim: int
    ) -> torch.Tensor:
        vision_outputs = self.vision_tower(pixel_values=pixel_values).last_hidden_state
        image_features = self.multi_modal_projector(
            vision_outputs, out_cols=proj_out_cols, final_embed_dim=final_embed_dim
        )
        return image_features

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ):
        # Same semantics as HF Gemma3
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(
                    self.config.image_token_id,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = (
            special_image_mask.unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        n_image_features = image_features.shape[0] * image_features.shape[1]
        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        return special_image_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # Dynamic dims
        current_num_heads: Optional[int] = None,
        dyn_embed_dim: Optional[int] = None,
        dyn_intermediate_size: Optional[int] = None,
        dyn_proj_out: Optional[int] = None,
        **lm_kwargs,
    ) -> Union[tuple, Gemma3ModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Replace image id with 0 (Gemma's pad token) if OOV to avoid index-errors (same as HF)
        if input_ids is not None and self.config.image_token_id >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)
            # If text model is dynamic, slice embeddings for text tokens:
            if dyn_embed_dim is not None and inputs_embeds.size(-1) > dyn_embed_dim:
                inputs_embeds = inputs_embeds[..., :dyn_embed_dim]

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        # Merge text and images
        image_features = None
        if pixel_values is not None:
            if dyn_proj_out is None or dyn_embed_dim is None:
                raise ValueError(
                    "dyn_proj_out and dyn_embed_dim must be provided for multimodal forwarding."
                )
            image_features = self.get_image_features(
                pixel_values, proj_out_cols=dyn_proj_out, final_embed_dim=dyn_embed_dim
            )
            image_features = image_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )

        # Build per-layer mask mapping
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config.get_text_config(),
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Detect "prefill" stage per HF logic
            is_prefill = (
                (not use_cache)
                or (past_key_values is None)
                or (not _cache_is_initialized_safe(past_key_values))
                or (pixel_values is not None)
            )
            if token_type_ids is not None and is_prefill:
                is_image = (token_type_ids == 1).to(cache_position.device)
                new_image_start = (
                    is_image & ~nn.functional.pad(is_image, (1, 0), value=0)[:, :-1]
                )
                image_group_ids = torch.cumsum(new_image_start.int(), dim=1) - 1
                image_group_ids = torch.where(
                    is_image,
                    image_group_ids,
                    torch.full_like(token_type_ids, -1, device=is_image.device),
                )
                mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
                    token_type_ids.to(cache_position.device),
                    image_group_ids,
                    self.config.mm_tokens_per_image,
                )

            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        outputs = self.language_model(
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            current_num_heads=current_num_heads,
            dyn_embed_dim=dyn_embed_dim,
            dyn_intermediate_size=dyn_intermediate_size,
            **lm_kwargs,
        )

        return Gemma3ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values if use_cache else None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


class FlexCoreGemma3ForConditionalGeneration(Gemma3PreTrainedModel, GenerationMixin):
    """
    Multimodal ConditionalGeneration head on top of FlexCoreGemma3Model.
    """

    # Map HF checkpoint keys to this module's hierarchy (same as HF Gemma3ForConditionalGeneration)
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^vision_tower": "model.vision_tower",
        "^multi_modal_projector": "model.multi_modal_projector",
        "^language_model.lm_head": "lm_head",
    }
    # we are filtering the logits/labels so we shouldn't divide the loss based on num_items_in_batch
    accepts_loss_kwargs = False
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Gemma3Config):
        super().__init__(config)
        self.model = FlexCoreGemma3Model(config)
        self.lm_head = DynamicLinear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def multi_modal_projector(self):
        return self.model.multi_modal_projector

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # Dynamic dims
        current_num_heads: Optional[int] = None,
        dyn_embed_dim: Optional[int] = None,
        dyn_intermediate_size: Optional[int] = None,
        dyn_proj_out: Optional[int] = None,
        **lm_kwargs,
    ) -> Union[tuple, Gemma3CausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=(
                return_dict if return_dict is not None else self.config.use_return_dict
            ),
            cache_position=cache_position,
            current_num_heads=current_num_heads,
            dyn_embed_dim=dyn_embed_dim,
            dyn_intermediate_size=dyn_intermediate_size,
            dyn_proj_out=dyn_proj_out,
            **lm_kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head.forward_sliced(
            hidden_states[:, slice_indices, :], in_cols=dyn_embed_dim
        )

        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(
                    logits.device
                )
                shift_logits = shift_logits[
                    shift_attention_mask.to(logits.device) != 0
                ].contiguous()
                shift_labels = shift_labels[
                    shift_attention_mask.to(shift_labels.device) != 0
                ].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            loss_fct = nn.CrossEntropyLoss()
            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)

        if not (
            return_dict if return_dict is not None else self.config.use_return_dict
        ):
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Gemma3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        attention_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        # Only pass pixel_values during prefill (cache_position[0] == 0); during cached decoding it must be None (same as HF)
        if cache_position is not None and cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
        return model_inputs

    @staticmethod
    def create_masks_for_generate(
        config: PretrainedConfig,
        input_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cache_position: torch.Tensor,
        past_key_values: Optional[Cache],
        position_ids: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        mask_kwargs = {
            "config": config.get_text_config(),
            "input_embeds": input_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        if token_type_ids is not None and input_embeds.shape[1] != 1:
            is_image = (token_type_ids == 1).to(cache_position.device)
            new_image_start = (
                is_image & ~nn.functional.pad(is_image, (1, 0), value=0)[:, :-1]
            )
            image_group_ids = torch.cumsum(new_image_start.int(), dim=1) - 1
            image_group_ids = torch.where(
                is_image, image_group_ids, torch.full_like(token_type_ids, -1)
            )
            mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
                token_type_ids.to(cache_position.device),
                image_group_ids,
                config.mm_tokens_per_image,
            )
        # return create_sliding_window_causal_mask(**mask_kwargs) if False else create_causal_mask(**mask_kwargs)
        return hf_create_masks_for_generate(**mask_kwargs)
