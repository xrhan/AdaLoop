"""fp32 master weights for the small, newly initialised modules (the loop core and the projector).

The loop core and the projector keep fp32 parameters while the LM and the CLIP tower run in bf16
(training) or fp16 (evaluation). :class:`FP32Pinned` keeps them fp32 under any later cast of a
parent module, and :func:`fp32_linear` / :func:`fp32_layer_norm` / :func:`fp32_rms_norm` compute
in fp32 regardless of how the parameters are currently stored.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def fp32_pinned_fn(fn):
    """Return an fp32-pinning replacement for ``fn`` if ``fn`` is a down-casting dtype conversion.

    Used by ``nn.Module._apply`` overrides (see :class:`FP32Pinned`): floating tensors take only
    the *device* effect of ``fn`` and stay in fp32 (tensors that were materialised in a lower
    precision are upcast, which is lossless). Returns ``None`` when ``fn`` does not down-cast,
    in which case the caller applies ``fn`` unchanged. Pinning must not round-trip a value
    through the lower precision, otherwise loaded fp32 weights would be silently rounded.
    """
    probe = fn(torch.zeros((), dtype=torch.float32))
    if not (probe.is_floating_point() and probe.dtype != torch.float32):
        return None

    def pinned(t):
        if not t.is_floating_point():
            return fn(t)
        dev = fn(torch.zeros((), dtype=torch.float32, device=t.device)).device
        return t if (t.device == dev and t.dtype == torch.float32) else t.to(device=dev, dtype=torch.float32)
    return pinned


class FP32Pinned:
    """Mixin: a down-casting ``.to()`` / ``.half()`` / ``.bfloat16()`` keeps floating parameters
    in fp32 (only the device changes).

    Pinning lives on the class rather than in a single ``.float()`` call so that any later cast of
    a parent module cannot undo it. It only governs storage; tensors materialised directly in a
    lower precision (e.g. by ``from_pretrained`` with a ``torch_dtype``) are upcast the next time
    a down-casting conversion is applied.
    """

    def _apply(self, fn, recurse=True):
        pinned = fp32_pinned_fn(fn)
        return super()._apply(pinned if pinned is not None else fn, recurse)


class FP32Module(FP32Pinned, nn.Module):
    """An ``nn.Module`` whose floating parameters stay fp32 under parent dtype casts."""


# Under bf16 autocast the matmul itself still follows autocast unless the caller disables it (see
# AdaLNCLIPEncoderLayer.forward and QuestionAttentionPooler.forward).
def fp32_linear(x, layer: nn.Linear):
    return F.linear(x.float(), layer.weight.float(), None if layer.bias is None else layer.bias.float())


def fp32_layer_norm(x, layer: nn.LayerNorm):
    return F.layer_norm(x.float(), layer.normalized_shape, layer.weight.float(), layer.bias.float(), layer.eps)


def fp32_rms_norm(x, layer: nn.RMSNorm):
    return F.rms_norm(x.float(), layer.normalized_shape, layer.weight.float(), layer.eps)
