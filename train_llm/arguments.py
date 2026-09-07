from typing import Dict
from transformers import TrainingArguments


def build_training_arguments(kwargs: Dict) -> TrainingArguments:
    """
    Create TrainingArguments robustly (compatible with slight API differences).
      - Handles eval_strategy/evaluation_strategy naming across versions.
    """
    allowed = set(getattr(TrainingArguments,
                  "__dataclass_fields__", {}).keys())

    eval_enabled = kwargs.pop("_eval_enabled", False)
    eval_steps = kwargs.pop("_eval_steps", None)
    if eval_enabled:
        if "eval_strategy" in allowed:
            kwargs["eval_strategy"] = "steps"
        elif "evaluation_strategy" in allowed:
            kwargs["evaluation_strategy"] = "steps"
        if "eval_steps" in allowed and eval_steps is not None:
            kwargs["eval_steps"] = eval_steps
    else:
        if "eval_strategy" in allowed:
            kwargs["eval_strategy"] = "no"
        elif "evaluation_strategy" in allowed:
            kwargs["evaluation_strategy"] = "no"

    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return TrainingArguments(**filtered)
