from typing import Optional
import torch
from transformers.utils import logging

# Use the flex wrapper as the core implementation
from .flex_gemma import FlexGemma3ForCausalLM, FlexGemma3ForConditionalGeneration

from transformers.models.gemma3.modeling_gemma3 import (
    Cache,
    CausalLMOutputWithPast,
    Gemma3PreTrainedModel,
)
from common_helpers import pick_best_attn_impl

logger = logging.get_logger(__name__)


class HydraGemma3ForCausalLM(FlexGemma3ForCausalLM):
    """
    Hydra behavior on top of Flex core (text-only):
      - scales by head ratio for embed/MLP
      - no extra width args for the user
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
        current_num_heads: Optional[int] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # Only pass hydra-style args (embed/MLP scale by head ratio)
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
            current_num_heads=current_num_heads,
            **kwargs,
        )

    # Match hydra_gemma_from_flex's GC toggles
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # Text-only: flag lives directly on self.model (FlexCoreGemma3TextModel)
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
    resolved_impl = pick_best_attn_impl(attn_implementation, device=device)
    if dtype is not None and "torch_dtype" not in model_kwargs:
        model_kwargs["torch_dtype"] = dtype

    model = HydraGemma3ForCausalLM.from_pretrained(
        model_name_or_path,
        attn_implementation=resolved_impl,
        **model_kwargs,
    )
    return model


# ----------------------------
# Multimodal ConditionalGeneration wrapper
# ----------------------------


class HydraGemma3ForConditionalGeneration(FlexGemma3ForConditionalGeneration):
    """
    Hydra behavior for multimodal:
      - text embed/MLP scale by head ratio
      - projection width also scales by the same head ratio
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
        logits_to_keep: int = 0,
        current_num_heads: Optional[int] = None,
        **kwargs,
    ):
        # Only pass hydra-style controls: scale by heads
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
            current_num_heads=current_num_heads,
            # No explicit embed_ratio/mlp_ratio/proj_ratio/width: all derived from head ratio internally
            **kwargs,
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # Fix Bug 1: call on the correct inner module (language_model = FlexCoreGemma3TextModel)
        # The flag self.gradient_checkpointing is checked in FlexCoreGemma3TextModel.forward()
        try:
            self.model.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        except TypeError:
            self.model.language_model.gradient_checkpointing_enable()

        # Also set on self for HF Trainer compatibility checks
        self.model.language_model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.model.language_model.gradient_checkpointing_disable()
        self.model.language_model.gradient_checkpointing = False

    # def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
    #     try:
    #         Gemma3PreTrainedModel.gradient_checkpointing_enable(
    #             self, gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    #     except TypeError:
    #         Gemma3PreTrainedModel.gradient_checkpointing_enable(self)
    #     self.model.gradient_checkpointing = True

    # def gradient_checkpointing_disable(self):
    #     Gemma3PreTrainedModel.gradient_checkpointing_disable(self)
    #     self.model.gradient_checkpointing = False


def get_mm_model(
    model_name_or_path: str,
    dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
    device: Optional[str] = None,
    **model_kwargs,
):
    """
    Load HydraGemma3ForConditionalGeneration (multimodal) based on flex.
    """
    resolved_impl = pick_best_attn_impl(attn_implementation, device=device)
    if dtype is not None and "torch_dtype" not in model_kwargs:
        model_kwargs["torch_dtype"] = dtype

    model = HydraGemma3ForConditionalGeneration.from_pretrained(
        model_name_or_path,
        attn_implementation=resolved_impl,
        **model_kwargs,
    )
    return model
