# MIT License
# Copyright (c) 2024 The HuggingFace Team
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import logging
import os
from typing import Union

from pydantic import Field, PositiveInt
from transformers import (
    AutoConfig,
    PretrainedConfig,
)
from pydantic import Field, PositiveInt
from typing import Union

from lighteval.models.abstract_model import ModelConfig
from lighteval.models.utils import _get_model_sha

from lighteval.models.custom.custom_model import CustomModelConfig


logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

STARTING_BATCH_SIZE = 512


class CustomTransformersModelConfig(CustomModelConfig, ModelConfig):
    """Configuration class for HuggingFace Transformers models.

    This configuration is used to load and configure models from the HuggingFace Transformers library.

    Attributes:
        model_name (str):
            HuggingFace Hub model ID or path to a pre-trained model. This corresponds to the
            `pretrained_model_name_or_path` argument in HuggingFace's `from_pretrained` method.
        tokenizer (str | None):
            Optional HuggingFace Hub tokenizer ID. If not specified, uses the same ID as model_name.
            Useful when the tokenizer is different from the model (e.g., for multilingual models).
        subfolder (str | None):
            Subfolder within the model repository. Used when models are stored in subdirectories.
        revision (str):
            Git revision of the model to load. Defaults to "main".
        batch_size (PositiveInt | None):
            Batch size for model inference. If None, will be automatically determined.
        max_length (PositiveInt | None):
            Maximum sequence length for the model. If None, uses model's default.
        model_loading_kwargs (dict):
            Additional keyword arguments passed to `from_pretrained`. Defaults to empty dict.
        add_special_tokens (bool):
            Whether to add special tokens during tokenization. Defaults to True.
        skip_special_tokens (bool):
            Whether the tokenizer should output special tokens back during generation. Needed for reasoning models. Defaults to True
        model_parallel (bool | None):
            Whether to use model parallelism across multiple GPUs. If None, automatically
            determined based on available GPUs and model size.
        dtype (str | None):
            Data type for model weights. Can be "float16", "bfloat16", "float32", "auto", "4bit", "8bit".
            If "auto", uses the model's default dtype.
        device (Union[int, str]):
            Device to load the model on. Can be "cuda", "cpu", or GPU index. Defaults to "cuda".
        trust_remote_code (bool):
            Whether to trust remote code when loading models. Defaults to False.
        compile (bool):
            Whether to compile the model using torch.compile for optimization. Defaults to False.
        multichoice_continuations_start_space (bool | None):
            Whether to add a space before multiple choice continuations. If None, uses model default.
            True forces adding space, False removes leading space if present.
        pairwise_tokenization (bool):
            Whether to tokenize context and continuation separately or together. Defaults to False.
        continuous_batching (bool):
            Whether to use continuous batching for generation. Defaults to False.
        override_chat_template (bool):
            If True, we force the model to use a chat template. If alse, we prevent the model from using
            a chat template. If None, we use the default (true if present in the tokenizer, false otherwise)
        generation_parameters (GenerationParameters, optional, defaults to empty GenerationParameters):
            Configuration parameters that control text generation behavior, including
            temperature, top_p, max_new_tokens, etc.
        system_prompt (str | None, optional, defaults to None): Optional system prompt to be used with chat models.
            This prompt sets the behavior and context for the model during evaluation.
        cache_dir (str, optional, defaults to "~/.cache/huggingface/lighteval"): Directory to cache the model.

    Example:
        ```python
        config = TransformersModelConfig(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            batch_size=4,
            dtype="float16",
            generation_parameters=GenerationParameters(
                temperature=0.7,
                max_new_tokens=100
            )
        )
        ```

    Note:
        This configuration supports quantization (4-bit and 8-bit) through the dtype parameter.
        When using quantization, ensure you have the required dependencies installed
        (bitsandbytes for 4-bit/8-bit quantization).
    """

    model_name: str
    tokenizer: str | None = None
    subfolder: str | None = None
    revision: str = "main"
    batch_size: PositiveInt | None = None
    max_length: PositiveInt | None = None
    model_loading_kwargs: dict = Field(default_factory=dict)
    add_special_tokens: bool = True
    skip_special_tokens: bool = True
    model_parallel: bool | None = None
    dtype: str | None = None
    device: Union[int, str] = "cuda"
    trust_remote_code: bool = False
    compile: bool = False
    multichoice_continuations_start_space: bool | None = None
    pairwise_tokenization: bool = False
    continuous_batching: bool = False
    override_chat_template: bool | None = None
    # e.g. "hydra_gemma" or "auto_model"
    implementation: str = "auto_model"
    # "auto" | "flash_attention_2" | "sdpa" | "eager"
    attn_implementation: str | None = None
    current_num_heads: int | None = None
    # MatFormer-style FFN control (maps to `mlp_ratio` in matformer_gemma)
    # A scalar applies globally; a list supplies one ratio per decoder layer.
    ffn_granularity_ratio: float | list[float] | None = None

    def model_post_init(self, __context):
        if self.multichoice_continuations_start_space is True:
            logger.warning(
                "You set `multichoice_continuations_start_space` to true. This will force multichoice continuations to use a starting space"
            )
        if self.multichoice_continuations_start_space is False:
            logger.warning(
                "You set `multichoice_continuations_start_space` to false. This will remove a leading space from multichoice continuations, if present."
            )

    def get_transformers_config(self) -> PretrainedConfig:
        revision = self.revision

        if self.subfolder:
            revision = f"{self.revision}/{self.subfolder}"

        auto_config = AutoConfig.from_pretrained(
            self.model_name,
            revision=revision,
            trust_remote_code=self.trust_remote_code,
        )

        return auto_config

    def get_model_sha(self):
        return _get_model_sha(repo_id=self.model_name, revision=self.revision)
