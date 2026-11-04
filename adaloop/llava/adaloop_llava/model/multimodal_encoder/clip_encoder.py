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
# llava/model/multimodal_encoder/clip_encoder.py: input-resolution retargeting and average pooling
# before the projector. The S2 tower, the CLS-token feature and other unused paths were removed.

"""CLIP vision tower with input-resolution retargeting and spatial average pooling."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

logger = logging.getLogger(__name__)


def retarget_resolution(vm, cfg, res: int, image_processor=None):
    """Change the input resolution of a CLIP vision model to ``res`` pixels.

    The patch-grid position embeddings are bicubically interpolated (``align_corners=False``,
    computed in fp32) to the new grid while the CLS position embedding is kept; the patch
    embedding convolution itself does not depend on the input size and is left untouched.
    ``num_patches``/``num_positions``/``position_ids`` and the image processor's resize and crop
    sizes are updated to match. ``res == 0`` is a no-op.
    """
    if not res:
        return
    emb = vm.embeddings
    p = int(emb.patch_size)
    if res % p:
        raise ValueError(f"Input resolution {res} is not divisible by the patch size {p}")
    g_new = res // p
    w = emb.position_embedding.weight.data
    d = w.shape[1]
    g_old = int(round((w.shape[0] - 1) ** 0.5))
    if g_old * g_old != w.shape[0] - 1:
        raise ValueError(f"Expected 1 + a square number of position embeddings, got {w.shape[0]}")
    if g_new != g_old:
        cls_w, patch_w = w[:1], w[1:]
        patch_w = patch_w.reshape(1, g_old, g_old, d).permute(0, 3, 1, 2).float()
        patch_w = F.interpolate(patch_w, size=(g_new, g_new), mode="bicubic", align_corners=False)
        patch_w = patch_w.permute(0, 2, 3, 1).reshape(g_new * g_new, d).to(w.dtype)
        new_w = torch.cat([cls_w, patch_w], dim=0)
        n_pos = new_w.shape[0]
        new_emb = nn.Embedding(n_pos, d, device=w.device, dtype=w.dtype)
        new_emb.weight.data.copy_(new_w)
        new_emb.weight.requires_grad_(False)
        emb.position_embedding = new_emb
        emb.num_patches = g_new * g_new
        emb.num_positions = n_pos
        emb.register_buffer("position_ids", torch.arange(n_pos, device=w.device).expand((1, -1)),
                            persistent=False)
        emb.image_size = res
        cfg.image_size = res
    if image_processor is not None:
        if getattr(image_processor, "crop_size", None):
            image_processor.crop_size["height"] = image_processor.crop_size["width"] = res
        if isinstance(getattr(image_processor, "size", None), dict):
            if "shortest_edge" in image_processor.size:
                image_processor.size["shortest_edge"] = res
            else:
                image_processor.size["height"] = image_processor.size["width"] = res


def spatial_pool(r, p: int):
    """Average-pool a square patch grid ``[B, G*G, D]`` by ``p x p`` to ``[B, (G/p)**2, D]``.

    Applied before the token-wise projector, so a projector trained on the pooled tokens can be
    reused unchanged.
    """
    p = int(p or 1)
    if p <= 1:
        return r
    b, n, d = r.shape
    g = int(round(n ** 0.5))
    if g * g != n:
        raise ValueError(f"spatial_pool needs a square patch grid, got {n} tokens")
    if g % p:
        raise ValueError(f"Patch grid {g}x{g} is not divisible by spatial_pool={p}")
    r = r.transpose(1, 2).reshape(b, d, g, g)
    r = F.avg_pool2d(r, kernel_size=p, stride=p)
    return r.reshape(b, d, -1).transpose(1, 2).contiguous()


class CLIPVisionTower(nn.Module):
    """Frozen CLIP vision tower: features of ``hidden_states[select_layer]`` without CLS, pooled.

    ``args`` is the model config (``LlavaConfig``); the geometry comes from
    ``mm_vision_select_layer``, ``vision_input_res`` and ``vision_spatial_pool``
    (``mm_vision_select_feature`` must be ``patch``).
    """

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        select_feature = getattr(args, "mm_vision_select_feature", "patch")
        if select_feature != "patch":
            raise ValueError(f"mm_vision_select_feature must be 'patch', got {select_feature!r}")
        self.input_res = int(getattr(args, "vision_input_res", 0) or 0)
        self.spatial_pool = int(getattr(args, "vision_spatial_pool", 1) or 1)

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            logger.info("%s is already loaded; skipping load_model.", self.vision_tower_name)
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)
        if self.input_res:
            retarget_resolution(self.vision_tower.vision_model, self.vision_tower.config,
                                self.input_res, self.image_processor)
            n_patches = self.vision_tower.vision_model.embeddings.num_patches
            logger.info("[vision] input resolution %dpx, %d patches, %dx%d pooling -> %d image tokens",
                        self.input_res, n_patches, self.spatial_pool, self.spatial_pool,
                        n_patches // (self.spatial_pool ** 2))

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer][:, 1:]
        return spatial_pool(image_features, self.spatial_pool)

    @torch.no_grad()
    def forward(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                               output_hidden_states=True)
        return self.feature_select(image_forward_outs).to(images.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size
