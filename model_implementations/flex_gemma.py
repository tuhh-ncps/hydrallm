from typing import List, Optional, Union
import torch
from transformers.utils import logging

from .modeling_gemma_flex import (
    FlexCoreGemma3ForCausalLM,
    FlexCoreGemma3ForConditionalGeneration,
)
from common_helpers import safe_int as _safe_int
from transformers.models.gemma3.modeling_gemma3 import (
    Cache,
    CausalLMOutputWithPast,
    Gemma3Config,
)
from common_helpers import pick_best_attn_impl

logger = logging.get_logger(__name__)


class FlexGemma3ForCausalLM(FlexCoreGemma3ForCausalLM):
    """
    A user-friendly wrapper for the FlexGemma3 core model (text-only).
    Provides high-level dynamic controls, translating user args into the explicit core dims.
    """

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
        # High-level dynamic width controls
        current_num_heads: Optional[int] = None,
        embed_width: Optional[int] = None,
        embed_ratio: Optional[float] = None,
        mlp_width: Optional[int] = None,
        mlp_ratio: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        base_heads = self.config.num_attention_heads
        final_num_heads = (
            current_num_heads if current_num_heads is not None else base_heads
        )
        head_ratio = (
            float(final_num_heads) / float(base_heads) if base_heads > 0 else 1.0
        )

        # dyn_embed_dim: priority explicit > ratio > head-based
        if embed_width is not None:
            dyn_embed_dim = _safe_int(embed_width)
        elif embed_ratio is not None:
            dyn_embed_dim = _safe_int(self.config.hidden_size * embed_ratio)
        else:
            dyn_embed_dim = _safe_int(self.config.hidden_size * head_ratio)

        # dyn_intermediate_size: priority explicit > ratio > head-based
        if mlp_width is not None:
            dyn_intermediate_size = _safe_int(mlp_width)
        elif mlp_ratio is not None:
            if isinstance(mlp_ratio, (list, tuple)):
                dyn_intermediate_size = [
                    _safe_int(self.config.intermediate_size * ratio)
                    for ratio in mlp_ratio
                ]
            else:
                dyn_intermediate_size = _safe_int(
                    self.config.intermediate_size * mlp_ratio
                )
        else:
            dyn_intermediate_size = _safe_int(
                self.config.intermediate_size * head_ratio
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            current_num_heads=final_num_heads,
            dyn_embed_dim=dyn_embed_dim,
            dyn_intermediate_size=dyn_intermediate_size,
            **kwargs,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        # Propagate dynamic width arguments
        dynamic_keys = [
            "current_num_heads",
            "embed_width",
            "embed_ratio",
            "mlp_width",
            "mlp_ratio",
        ]
        for key in dynamic_keys:
            if key in kwargs:
                model_inputs[key] = kwargs[key]
        return model_inputs


def get_model(
    model_name_or_path: str,
    dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
    device: Optional[str] = None,
    **model_kwargs,
):
    """
    Load the user-facing FlexGemma3ForCausalLM with optional from_pretrained kwargs.
    """
    resolved_impl = pick_best_attn_impl(attn_implementation, device=device)
    if dtype is not None and "torch_dtype" not in model_kwargs:
        model_kwargs["torch_dtype"] = dtype

    model = FlexGemma3ForCausalLM.from_pretrained(
        model_name_or_path,
        attn_implementation=resolved_impl,
        **model_kwargs,
    )
    return model


# ----------------------------
# Multimodal ConditionalGeneration wrapper
# ----------------------------


class FlexGemma3ForConditionalGeneration(FlexCoreGemma3ForConditionalGeneration):
    """
    Multimodal wrapper with convenient controls:
      - current_num_heads, embed_width/embed_ratio, mlp_width/mlp_ratio (text dynamic dims)
      - proj_width/proj_ratio: controls projection output width; by default equals dyn_embed_dim
    """

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
        # Dynamic dims (text)
        current_num_heads: Optional[int] = None,
        embed_width: Optional[int] = None,
        embed_ratio: Optional[float] = None,
        mlp_width: Optional[int] = None,
        mlp_ratio: Optional[Union[float, List[float]]] = None,
        # Dynamic projector control
        proj_width: Optional[int] = None,
        proj_ratio: Optional[float] = None,
        **kwargs,
    ):
        base_heads = self.config.text_config.num_attention_heads
        final_num_heads = (
            current_num_heads if current_num_heads is not None else base_heads
        )
        head_ratio = (
            float(final_num_heads) / float(base_heads) if base_heads > 0 else 1.0
        )

        # dyn_embed_dim for text tokens
        if embed_width is not None:
            dyn_embed_dim = _safe_int(embed_width)
        elif embed_ratio is not None:
            dyn_embed_dim = _safe_int(self.config.text_config.hidden_size * embed_ratio)
        else:
            dyn_embed_dim = _safe_int(self.config.text_config.hidden_size * head_ratio)

        # dyn_intermediate_size for MLP
        if mlp_width is not None:
            dyn_intermediate_size = _safe_int(mlp_width)
        elif mlp_ratio is not None:
            if isinstance(mlp_ratio, (list, tuple)):
                dyn_intermediate_size = [
                    _safe_int(self.config.text_config.intermediate_size * ratio)
                    for ratio in mlp_ratio
                ]
            else:
                dyn_intermediate_size = _safe_int(
                    self.config.text_config.intermediate_size * mlp_ratio
                )
        else:
            dyn_intermediate_size = _safe_int(
                self.config.text_config.intermediate_size * head_ratio
            )

        # dyn_proj_out for projector (defaults to dyn_embed_dim)
        if proj_width is not None:
            dyn_proj_out = _safe_int(proj_width)
        elif proj_ratio is not None:
            dyn_proj_out = _safe_int(self.config.text_config.hidden_size * proj_ratio)
        else:
            dyn_proj_out = dyn_embed_dim

        # Fail early if using images with a non-positive projector width
        if pixel_values is not None and dyn_proj_out <= 0:
            raise ValueError("Invalid dyn_proj_out for projector.")

        return super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            token_type_ids=token_type_ids,
            cache_position=cache_position,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            logits_to_keep=logits_to_keep,
            current_num_heads=final_num_heads,
            dyn_embed_dim=dyn_embed_dim,
            dyn_intermediate_size=dyn_intermediate_size,
            dyn_proj_out=dyn_proj_out,
            **kwargs,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        # propagate dynamic controls (and token_type_ids) across generation
        dynamic_keys = [
            "current_num_heads",
            "embed_width",
            "embed_ratio",
            "mlp_width",
            "mlp_ratio",
            "proj_width",
            "proj_ratio",
            "token_type_ids",
        ]
        for key in dynamic_keys:
            if key in kwargs:
                model_inputs[key] = kwargs[key]
        return model_inputs


def get_mm_model(
    model_name_or_path: str,
    dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
    device: Optional[str] = None,
    **model_kwargs,
):
    """
    Load the user-facing FlexGemma3ForConditionalGeneration (multimodal) with optional from_pretrained kwargs.
    """
    resolved_impl = pick_best_attn_impl(attn_implementation, device=device)
    if dtype is not None and "torch_dtype" not in model_kwargs:
        model_kwargs["torch_dtype"] = dtype

    model = FlexGemma3ForConditionalGeneration.from_pretrained(
        model_name_or_path,
        attn_implementation=resolved_impl,
        **model_kwargs,
    )
    return model
