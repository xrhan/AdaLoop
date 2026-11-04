"""Batched greedy generation, shared by the pipeline (:mod:`adaloop_llava.inference`) and the VisRO
evaluation (:func:`adaloop_llava.eval.generate.run_depth`).

A query is a dict with ``question`` (the bare question), an optional ``prompt`` (the
language-model text; default: the question) and ``image`` (an image path, encoded bytes, or an
:class:`adaloop_common.data.ImageSource`). For one batch of queries at one depth:

* prompt: :func:`~adaloop_llava.conversation.vicuna_v1_prompt` with ``"<image>\\n" + prompt.strip()``
  as the user turn and the assistant turn left open; token ids from :func:`tokenizer_image_token`;
* the prompts are left-padded by hand, and ``model.config.tokenizer_padding_side`` is set to
  ``"left"`` for the duration of the call: the multimodal splice re-pads the sequences according
  to that setting, and right padding would make the shorter prompts of a batch generate after
  their padding;
* question condition (loop models): :func:`tokenize_question_condition` (BOS + question tokens,
  at most ``condition_max_length``), right-padded with a length mask;
* images: RGB, padded to a square with the CLIP mean colour, CLIP preprocessing
  (:func:`~adaloop_llava.mm_utils.process_image`), cast to the dtype of the model's first
  parameter (the evaluation dtype);
* ``generate(do_sample=False, max_new_tokens, use_cache=True)``; loop models also get the
  condition and ``loop_steps = depth``.
"""

from __future__ import annotations

import io
from typing import Any, Dict, Sequence

import torch
from PIL import Image

from .constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from .conversation import vicuna_v1_prompt
from .data.dataset import tokenize_question_condition
from .mm_utils import process_image, tokenizer_image_token


def _load_image(source: Any) -> Image.Image:
    """An image path, encoded bytes or an object with ``read_bytes()``, as RGB."""
    if hasattr(source, "read_bytes") and not isinstance(source, str):
        source = io.BytesIO(source.read_bytes())
    elif isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    with Image.open(source) as im:
        return im.convert("RGB")


def make_batch(queries: Sequence[Dict[str, Any]], tokenizer, image_processor, condition_max_length: int = 128):
    """Left-padded prompt ids and mask, right-padded question condition and mask, and pixels.

    The language-model prompt is built from ``query["prompt"]`` when the query has one, else from
    ``query["question"]``; the question condition always comes from ``query["question"]``."""
    prompts, conds, imgs = [], [], []
    for d in queries:
        prompts.append(vicuna_v1_prompt(DEFAULT_IMAGE_TOKEN + "\n" + d.get("prompt", d["question"]).strip()))
        conds.append(tokenize_question_condition(tokenizer, d["question"], condition_max_length))
        imgs.append(_load_image(d["image"]))
    ids = [tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for p in prompts]
    L = max(x.shape[0] for x in ids)
    input_ids = torch.full((len(ids), L), tokenizer.pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(ids), L), dtype=torch.bool)
    for j, x in enumerate(ids):
        input_ids[j, L - x.shape[0]:] = x
        attn[j, L - x.shape[0]:] = True
    Lc = max(c.shape[0] for c in conds)
    cond_ids = torch.full((len(conds), Lc), tokenizer.pad_token_id, dtype=torch.long)
    cond_mask = torch.zeros((len(conds), Lc), dtype=torch.bool)
    for j, c in enumerate(conds):
        cond_ids[j, :c.shape[0]] = c
        cond_mask[j, :c.shape[0]] = True
    pix = torch.stack([process_image(im, image_processor) for im in imgs])
    return input_ids, attn, cond_ids, cond_mask, pix


def generate_ids(model, tokenizer, image_processor, queries: Sequence[Dict[str, Any]], depth: int, *,
                 is_loop: bool, device, max_new_tokens: int = 32, condition_max_length: int = 128) -> torch.Tensor:
    """Greedy generated token ids (``model.generate`` output) for one batch of queries at one depth."""
    input_ids, attn, cond_ids, cond_mask, pix = make_batch(queries, tokenizer, image_processor,
                                                           condition_max_length)
    pix = pix.to(device=device, dtype=next(model.parameters()).dtype)
    # Only loop models accept the condition and the depth; a plain model would pass them on to
    # the language model's forward.
    loop_kw = {} if not is_loop else dict(
        condition_input_ids=cond_ids.to(device), condition_attention_mask=cond_mask.to(device),
        loop_steps=depth)
    side = getattr(model.config, "tokenizer_padding_side", "right")
    model.config.tokenizer_padding_side = "left"
    try:
        with torch.inference_mode():
            return model.generate(inputs=input_ids.to(device), attention_mask=attn.to(device), images=pix,
                                  do_sample=False, max_new_tokens=max_new_tokens, use_cache=True, **loop_kw)
    finally:
        model.config.tokenizer_padding_side = side
