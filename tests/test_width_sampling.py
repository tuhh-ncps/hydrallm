from types import SimpleNamespace
from unittest.mock import patch

import pytest

from train_llm.trainer_matformer import MatFormerTrainer
from common_helpers import normalize_mlp_ratio
from model_implementations.matformer_gemma import (
    MatFormerGemma3ForConditionalGeneration,
)


def make_trainer(mode="global"):
    trainer = MatFormerTrainer.__new__(MatFormerTrainer)
    trainer.ffn_granularity_ratios = [0.25, 0.5, 1.0]
    trainer.ffn_granularity_weights = None
    trainer.width_sampling_mode = mode
    return trainer


def test_global_width_sampling_picks_one_granularity():
    trainer = make_trainer()
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=4))

    with patch("train_llm.trainer_matformer.random.choice", return_value=1) as choice:
        sampled = trainer._pick_granularity(model)

    assert sampled == 1
    choice.assert_called_once()


def test_per_layer_width_sampling_picks_each_layer_independently():
    trainer = make_trainer("per_layer")
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=4))

    with patch(
        "train_llm.trainer_matformer.random.choice", side_effect=[0, 2, 1, 0]
    ) as choice:
        sampled = trainer._pick_granularity(model)

    assert sampled == [0, 2, 1, 0]
    assert choice.call_count == 4
    assert trainer._effective_g(sampled) == "per_layer"


def test_per_layer_width_sampling_requires_layer_count():
    trainer = make_trainer("per_layer")
    model = SimpleNamespace(config=SimpleNamespace())

    with pytest.raises(ValueError, match="num_hidden_layers"):
        trainer._pick_granularity(model)


def test_inference_ratio_defaults_to_global_for_one_value():
    assert normalize_mlp_ratio([0.5], num_hidden_layers=4) == 0.5


def test_inference_ratio_accepts_one_value_per_layer():
    assert normalize_mlp_ratio(
        [0.25, 0.5, 0.75, 1.0], num_hidden_layers=4
    ) == [0.25, 0.5, 0.75, 1.0]


def test_inference_ratio_rejects_wrong_layer_count():
    with pytest.raises(ValueError, match="expected 4, got 2"):
        normalize_mlp_ratio([0.5, 1.0], num_hidden_layers=4)


def test_matformer_multimodal_does_not_tie_projector_to_ffn_ratio():
    model = MatFormerGemma3ForConditionalGeneration.__new__(
        MatFormerGemma3ForConditionalGeneration
    )
    model.ffn_granularity_ratios = [0.25, 0.5, 1.0]

    with patch.object(
        MatFormerGemma3ForConditionalGeneration.__mro__[1],
        "forward",
        return_value="output",
    ) as forward:
        output = model.forward(granularity=1)

    assert output == "output"
    kwargs = forward.call_args.kwargs
    assert kwargs["embed_ratio"] == 1.0
    assert kwargs["mlp_ratio"] == 0.5
    assert "proj_ratio" not in kwargs
