"""Public package exports.

Keep model imports lazy so data-preparation utilities can import lite_llm
without requiring torch in tokenizer-only environments.
"""

from .configuration import LiteLlmConfig


def __getattr__(name):
    if name == "LiteLlmForCausalLM":
        from .modeling import LiteLlmForCausalLM

        return LiteLlmForCausalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["LiteLlmConfig", "LiteLlmForCausalLM"]
