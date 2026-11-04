# Copyright 2023 Haotian Liu
# Copyright 2026 The AdaLoop Authors.
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
#
# Modified by the AdaLoop Authors from LLaVA (https://github.com/haotian-liu/LLaVA)
# llava/model/multimodal_projector/builder.py: the mlp2x_gelu projector keeps fp32 master weights
# and computes in fp32; other projector types were removed.

"""The mlp2x_gelu vision projector with fp32 master weights."""

import torch.nn as nn

from ..fp32 import FP32Pinned, fp32_linear


class FP32MLPProjector(FP32Pinned, nn.Sequential):
    """``mlpNx_gelu`` projector with fp32-pinned parameters and fp32 linear layers.

    The state-dict keys are those of ``nn.Sequential`` (``0.weight``, ``0.bias``, ``2.weight``,
    ``2.bias``), so upstream LLaVA projector weights load unchanged. The output is returned in
    the input dtype; ``prepare_inputs_labels_for_multimodal`` then casts it to the embedding dtype.
    """

    def forward(self, x):
        out_dtype = x.dtype
        h = x.float()
        for layer in self:
            h = fp32_linear(h, layer) if isinstance(layer, nn.Linear) else layer(h)
        return h.to(out_dtype)


def build_vision_projector(config):
    """Linear(mm_hidden, hidden) -> GELU -> Linear(hidden, hidden), as in LLaVA-1.5."""
    projector_type = getattr(config, "mm_projector_type", None)
    if projector_type != "mlp2x_gelu":
        raise ValueError(f"Only the 'mlp2x_gelu' projector is supported, got {projector_type!r}")
    return FP32MLPProjector(nn.Linear(config.mm_hidden_size, config.hidden_size), nn.GELU(),
                            nn.Linear(config.hidden_size, config.hidden_size))
