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
# llava/model/multimodal_encoder/builder.py: the tower type is chosen by the loop mode of the
# model config.

"""Vision tower factory: plain CLIP or the looped AdaLN tower, by loop mode."""

from ..loop_utils import infer_loop_mode
from .clip_encoder import CLIPVisionTower
from .looped_clip import LoopedCLIPVisionTower


def build_vision_tower(config, delay_load=False):
    """Build the vision tower described by a model config.

    ``loop_mode == 'none'`` gives the plain :class:`CLIPVisionTower`; ``'single_pass'`` and
    ``'adaloop'`` give a :class:`LoopedCLIPVisionTower` whose loop spec is read from the same
    config. Geometry (``vision_input_res``, ``vision_spatial_pool``, ``mm_vision_select_layer``)
    always comes from the config. The same function serves training (after
    ``initialize_vision_modules`` wrote the keys) and ``from_pretrained`` at evaluation time.
    """
    vision_tower = getattr(config, "mm_vision_tower", None)
    if not vision_tower:
        raise ValueError("The model config has no mm_vision_tower")
    if infer_loop_mode(config) == "none":
        return CLIPVisionTower(vision_tower, args=config, delay_load=delay_load)
    return LoopedCLIPVisionTower(vision_tower, args=config, delay_load=delay_load)
