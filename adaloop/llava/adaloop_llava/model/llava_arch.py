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
# llava/model/llava_arch.py: the loop core is built and registered on the model, the vision tower
# is chosen by the loop mode, precomputed image features can be passed in, and the projector
# output is cast to the embedding dtype. Unused merge modes, image sizes and stage-1 code were
# removed. The upstream split into these mixins and llava_llama.py is kept so that both files
# stay comparable with upstream LLaVA.

"""LLaVA multimodal architecture: vision modules, the loop core and image-token splicing."""

from abc import ABC, abstractmethod

import torch

from ..constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from .multimodal_encoder.looped_clip import LoopCore
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector


def check_images(images):
    """Images enter the model as one ``[B, 3, H, W]`` tensor (one image per example) or ``None``."""
    if images is not None and not (torch.is_tensor(images) and images.ndim == 4):
        raise ValueError("Pass images as one [B, 3, H, W] tensor (one image per example)")


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            tower = build_vision_tower(config, delay_load=True)
            # The loop core is registered *before* the tower. named_parameters() keeps the first
            # name of a shared parameter, so the core's tensors are always named model.loop_core.*
            # (the tower adopts the same modules), which is how checkpoints store them.
            if getattr(tower, "loop_spec", None) is not None:
                self.loop_core = LoopCore(tower.loop_spec, vision_width=config.mm_hidden_size)
                tower.attach_loop_core(self.loop_core)
            self.vision_tower = tower
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        return getattr(self, "vision_tower", None)

    def initialize_vision_modules(self, cfg):
        """Set up the vision side for training from a release config (``adaloop_llava.config.Config``).

        * Writes the vision / loop settings into ``self.config`` (they end up in config.json, and
          ``from_pretrained`` rebuilds the same tower from them).
        * ``model.loop.mode == 'none'``: reuses the plain tower built from the base config (its
          geometry must match the release config), or builds one.
        * Loop modes: replaces any plain tower, builds and registers the loop core, attaches it,
          then loads CLIP (which retargets the resolution and adopts the core's modules).
        * The projector is made trainable and upcast to fp32. When it came from a bf16
          ``from_pretrained`` it was materialised in bf16, so training starts from the
          bf16-rounded base projector held in fp32.
        """
        from ..config import model_config_updates

        updates = model_config_updates(cfg)
        for k, v in updates.items():
            setattr(self.config, k, v)
        self.config.lm_hidden_size = self.config.hidden_size
        mode = updates["loop_mode"]

        existing = self.get_vision_tower()
        if mode != "none":
            if getattr(self, "loop_core", None) is not None:
                raise ValueError("The base model already contains a loop core; train from a plain LLaVA base")
            if existing is not None:
                # Unregister first: re-assigning an existing attribute keeps its registration slot,
                # which would place the tower before loop_core and rename the core's parameters.
                del self._modules["vision_tower"]
            vision_tower = build_vision_tower(self.config, delay_load=True)
            self.loop_core = LoopCore(vision_tower.loop_spec, vision_width=vision_tower.hidden_size)
            vision_tower.attach_loop_core(self.loop_core)
            vision_tower.load_model()
            self.vision_tower = vision_tower
        elif existing is None:
            vision_tower = build_vision_tower(self.config, delay_load=True)
            vision_tower.load_model()
            self.vision_tower = vision_tower
        else:
            vision_tower = existing
            vision_tower.load_model()
            if getattr(vision_tower, "loop_spec", None) is not None:
                raise ValueError("loop mode 'none' but the base model carries a loop tower")
            want = (updates["vision_input_res"], updates["vision_spatial_pool"], updates["mm_vision_select_layer"])
            got = (vision_tower.input_res, vision_tower.spatial_pool, vision_tower.select_layer)
            if tuple(int(x) for x in want) != tuple(int(x) for x in got):
                raise ValueError(f"The reused vision tower has (input_res, spatial_pool, select_layer)={got}, "
                                 f"but the config asks for {want}")

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_tower.hidden_size

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            for p in self.mm_projector.parameters():
                p.requires_grad = True
            self.mm_projector.float()


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_features=None
    ):
        """Splice projected image features into the token embeddings at ``IMAGE_TOKEN_INDEX``.

        ``image_features`` (already projected, ``[B, N, D]``) can be supplied to splice a
        precomputed readout, e.g. one per loop iteration for deep supervision; otherwise the
        images are encoded here. A list of ``[N, D]`` tensors fills the ``<image>`` tokens of the
        batch in row-major order (Variant C, whose rows hold one ``<image>`` per round).
        """
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if image_features is None:
            image_features = self.encode_images(images)

        # The projector computes in fp32; cast its output to the LM embedding dtype here, where
        # every readout path converges. Otherwise torch.cat would silently promote the whole
        # residual stream of the LM to fp32.
        emb_dtype = self.get_model().embed_tokens.weight.dtype
        if type(image_features) is list:
            image_features = [x.to(emb_dtype) for x in image_features]
        else:
            image_features = image_features.to(emb_dtype)

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # Remove the padding using attention_mask.
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer.
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them.
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
