from typing import List, Optional, Union
import torch
from transformers.utils import logging

from .flex_gemma import FlexGemma3ForCausalLM, FlexGemma3ForConditionalGeneration
from common_helpers import pick_best_attn_impl

logger = logging.get_logger(__name__)

class MatFormerGemma3ForCausalLM(FlexGemma3ForCausalLM):
    """
    MatFormer-style wrapper (text-only):
      - Keep embedding width at full
      - Vary FFN width via granularity or explicit mlp_ratio
    """

    def __init__(self, config, **kwargs):
        super().__init__(config)
        default_ratios = [0.25, 0.5, 0.75, 1.0]
        self.ffn_granularity_ratios = kwargs.get(
            "ffn_granularity_ratios", default_ratios)
        self.config.ffn_granularity_ratios = self.ffn_granularity_ratios

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # High-level MatFormer control
        granularity: Optional[Union[int, List[int]]] = None,
        mlp_ratio: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ):
        final_mlp_ratio = None
        if mlp_ratio is not None:
            final_mlp_ratio = mlp_ratio
        elif granularity is not None:
            granularities = (
                granularity if isinstance(granularity, (list, tuple)) else [granularity]
            )
            if any(
                not (0 <= value < len(self.ffn_granularity_ratios))
                for value in granularities
            ):
                raise ValueError(
                    f"Granularity must be between 0 and {len(self.ffn_granularity_ratios) - 1}"
                )
            ratios = [self.ffn_granularity_ratios[value] for value in granularities]
            final_mlp_ratio = (
                ratios if isinstance(granularity, (list, tuple)) else ratios[0]
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
            # MatFormer settings:
            embed_ratio=1.0,                # full embedding width
            current_num_heads=None,         # full number of heads
            mlp_ratio=final_mlp_ratio,      # FFN width varies
            # -> actually only mlp_ratio would be correct? TODO:
            **kwargs,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        for key in ["granularity", "mlp_ratio"]:
            if key in kwargs:
                model_inputs[key] = kwargs[key]
        return model_inputs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        try:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        except TypeError:
            self.model.gradient_checkpointing_enable()
        self.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()
        self.model.gradient_checkpointing = False


def get_model(
    model_name_or_path: str,
    dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
    device: Optional[str] = None,
    **model_kwargs,
):
    """
    Load the user-facing MatFormerGemma3ForCausalLM with optional from_pretrained kwargs.
    """
    resolved_impl = pick_best_attn_impl(attn_implementation, device=device)
    if dtype is not None and "torch_dtype" not in model_kwargs:
        model_kwargs["torch_dtype"] = dtype

    model = MatFormerGemma3ForCausalLM.from_pretrained(
        model_name_or_path,
        attn_implementation=resolved_impl,
        **model_kwargs,
    )
    return model


# ----------------------------
# Multimodal ConditionalGeneration wrapper
# ----------------------------

class MatFormerGemma3ForConditionalGeneration(FlexGemma3ForConditionalGeneration):
    """
    MatFormer-style wrapper (multimodal):
      - Keep embedding width at full
      - Vary FFN width via granularity or explicit mlp_ratio
      - Keep projection width aligned with the full embedding width
    """

    def __init__(self, config, **kwargs):
        super().__init__(config)
        default_ratios = [0.25, 0.5, 0.75, 1.0]
        self.ffn_granularity_ratios = kwargs.get(
            "ffn_granularity_ratios", default_ratios)
        self.config.ffn_granularity_ratios = self.ffn_granularity_ratios

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        granularity: Optional[Union[int, List[int]]] = None,
        mlp_ratio: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ):
        final_mlp_ratio = None
        if mlp_ratio is not None:
            final_mlp_ratio = mlp_ratio
        elif granularity is not None:
            granularities = (
                granularity if isinstance(granularity, (list, tuple)) else [granularity]
            )
            if any(
                not (0 <= value < len(self.ffn_granularity_ratios))
                for value in granularities
            ):
                raise ValueError(
                    f"Granularity must be between 0 and {len(self.ffn_granularity_ratios) - 1}"
                )
            ratios = [self.ffn_granularity_ratios[value] for value in granularities]
            final_mlp_ratio = (
                ratios if isinstance(granularity, (list, tuple)) else ratios[0]
            )

        # Embed and projector widths stay full; only the FFN width varies.
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
            current_num_heads=None,          # full heads
            embed_ratio=1.0,                 # full embedding width
            mlp_ratio=final_mlp_ratio,       # FFN varies
            **kwargs,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        for key in ["granularity", "mlp_ratio"]:
            if key in kwargs:
                model_inputs[key] = kwargs[key]
        return model_inputs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        try:
            self.model.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        except TypeError:
            self.model.language_model.gradient_checkpointing_enable()
        self.model.language_model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.model.language_model.gradient_checkpointing_disable()
        self.model.language_model.gradient_checkpointing = False


def get_mm_model(
    model_name_or_path: str,
    dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
    device: Optional[str] = None,
    **model_kwargs,
):
    """
    Load the user-facing MatFormerGemma3ForConditionalGeneration with optional from_pretrained kwargs.
    """
    resolved_impl = pick_best_attn_impl(attn_implementation, device=device)
    if dtype is not None and "torch_dtype" not in model_kwargs:
        model_kwargs["torch_dtype"] = dtype

    model = MatFormerGemma3ForConditionalGeneration.from_pretrained(
        model_name_or_path,
        attn_implementation=resolved_impl,
        **model_kwargs,
    )
    return model
