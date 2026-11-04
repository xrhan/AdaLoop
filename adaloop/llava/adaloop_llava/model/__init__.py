"""Model package. ``LlavaConfig`` / ``LlavaLlamaForCausalLM`` are imported on first access so that
light-weight helpers (e.g. :mod:`adaloop_llava.model.loop_utils`, used by the config parser) do
not pull in torch and transformers."""

__all__ = ["LlavaConfig", "LlavaLlamaForCausalLM"]


def __getattr__(name):
    if name in __all__:
        from . import llava_llama
        return getattr(llava_llama, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
