# Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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
# llava/train/train.py, which adopts code from FastChat (https://github.com/lm-sys/FastChat) and
# Stanford Alpaca (https://github.com/tatsu-lab/stanford_alpaca). Only the path LLaVA-1.5 takes for
# one image question-answer turn is kept: the image placeholder normalisation and the Vicuna-v1
# tokenisation with answer-only labels.

"""One image question-answer turn -> ``input_ids`` / ``labels`` with the Vicuna-v1 template."""

import logging
from typing import Tuple

import tokenizers
import torch
from packaging import version

from ..constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from ..conversation import ASSISTANT, SEP, SEP2, vicuna_v1_prompt
from ..mm_utils import tokenizer_image_token

logger = logging.getLogger(__name__)

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")


def tokenize_image_qa(tokenizer, question: str, answer: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render one turn with the Vicuna-v1 template and tokenise it; returns ``(input_ids, labels)``.

    Prompt: ``"<system> USER: <image>\\n<question> ASSISTANT: <answer></s>"`` with the question and
    the answer stripped; an ``<image>`` placeholder already in the question is moved to the front.
    Labels are ``IGNORE_INDEX`` everywhere except on the answer and its ``</s>``. If the per-round
    token count does not add up to the tokenised conversation, the whole sample is masked (and a
    warning is logged), as in upstream LLaVA.
    """
    user = (DEFAULT_IMAGE_TOKEN + "\n" + question.replace(DEFAULT_IMAGE_TOKEN, "").strip()).strip()
    conversation = vicuna_v1_prompt(user, answer.strip())
    input_ids = tokenizer_image_token(conversation, tokenizer, return_tensors="pt")
    target = input_ids.clone()

    # Mask targets
    sep = SEP + ASSISTANT + ": "
    total_len = int(target.ne(tokenizer.pad_token_id).sum())

    rounds = conversation.split(SEP2)
    cur_len = 1
    target[:cur_len] = IGNORE_INDEX
    for i, rou in enumerate(rounds):
        if rou == "":
            break

        parts = rou.split(sep)
        if len(parts) != 2:
            break
        parts[0] += sep

        round_len = len(tokenizer_image_token(rou, tokenizer))
        instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2

        if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
            round_len -= 1
            instruction_len -= 1

        target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

        cur_len += round_len
    target[cur_len:] = IGNORE_INDEX

    if cur_len < tokenizer.model_max_length:
        if cur_len != total_len:
            target[:] = IGNORE_INDEX
            logger.warning("tokenization mismatch: %d vs. %d (sample ignored)", cur_len, total_len)

    return input_ids, target
