"""LLaVA training samples: one VisRO or replay example -> one LLaVA sample plus the question condition.

* Examples: :func:`adaloop_common.data.load_training_set`, the VisRO training rows of the recipe's
  tasks followed by the replay rows, pooled with equal row weight.
* Supervision target: the short answer, stripped.
* Conversation: one turn, ``USER: <image>\\n{question} ASSISTANT: {answer}</s>`` (Vicuna v1,
  :func:`~adaloop_llava.data.preprocess.tokenize_image_qa`); the loss covers the answer tokens and
  ``</s>`` only.
* Question condition (input of the loop tower's question pooler): the question tokens without
  special tokens, BOS prepended, truncated to ``condition_max_length`` (BOS + up to 127 tokens
  for the default 128). It is produced for every setting; models without a loop ignore it.
* Image: RGB, padded to a square with the CLIP mean colour, then the CLIP image processor
  (whose resize / crop size the tower retargets to the training resolution, 448px;
  :func:`~adaloop_llava.mm_utils.process_image`).
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from adaloop_common.data import ImageQAExample

from ..constants import IGNORE_INDEX
from ..mm_utils import process_image
from .preprocess import tokenize_image_qa

logger = logging.getLogger(__name__)


def tokenize_question_condition(tokenizer, question: str, max_length: int = 128) -> torch.Tensor:
    """Question tokens (no special tokens) with BOS prepended, truncated to ``max_length``."""
    if max_length < 1:
        raise ValueError("condition_max_length must be >= 1")
    ids = tokenizer.encode(question, add_special_tokens=False)
    bos = tokenizer.bos_token_id
    if bos is not None and (not ids or ids[0] != bos):
        ids = [bos, *ids]
    ids = ids[:max_length]
    if not ids:
        raise ValueError("Question condition tokenized to an empty sequence")
    return torch.tensor(ids, dtype=torch.long)


class TrainingDataset(Dataset):
    """Pooled training examples as LLaVA training samples.

    Args:
        examples: the pooled examples (:attr:`adaloop_common.data.TrainingSet.examples`).
        tokenizer: the LLaVA tokenizer (slow Llama tokenizer, ``pad_token = unk_token``).
        image_processor: the vision tower's (retargeted) ``CLIPImageProcessor``.
        condition_max_length: length cap of the question condition, including BOS.
        sample_total: if set, keep ``sample_total`` examples evenly spaced over the pooled list
            (deterministic; every source keeps its share).
    """

    def __init__(self, examples: Sequence[ImageQAExample], tokenizer, image_processor, *,
                 condition_max_length: int = 128, sample_total: Optional[int] = None):
        super().__init__()
        self.examples = list(examples)
        if not self.examples:
            raise ValueError("TrainingDataset needs at least one example")
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.condition_max_length = int(condition_max_length)
        if sample_total and int(sample_total) < len(self.examples):
            n, m = int(sample_total), len(self.examples)
            self.examples = [self.examples[i * m // n] for i in range(n)]
            logger.info("[data] data_limit: %d -> %d examples (evenly spaced)", m, n)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        ex = self.examples[i]
        with Image.open(io.BytesIO(ex.image.read_bytes())) as im:
            image = process_image(im.convert("RGB"), self.image_processor)
        input_ids, labels = tokenize_image_qa(self.tokenizer, ex.question, ex.answer)
        return dict(input_ids=input_ids, labels=labels, image=image,
                    condition_input_ids=tokenize_question_condition(self.tokenizer, ex.question,
                                                                    self.condition_max_length))


@dataclass
class TrainingCollator:
    """Right-pads ``input_ids`` (with the pad id) and ``labels`` (with ``IGNORE_INDEX``), truncates
    both to ``tokenizer.model_max_length``, stacks the images, and right-pads the question
    condition with a mask built from the true lengths."""
    tokenizer: Any

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence([x["input_ids"] for x in instances], batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence([x["labels"] for x in instances], batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        L = self.tokenizer.model_max_length
        batch = dict(input_ids=input_ids[:, :L], labels=labels[:, :L],
                     attention_mask=input_ids[:, :L].ne(self.tokenizer.pad_token_id))
        batch["images"] = torch.stack([x["image"] for x in instances])
        cond = [x["condition_input_ids"] for x in instances]
        batch["condition_input_ids"] = torch.nn.utils.rnn.pad_sequence(cond, batch_first=True,
                                                                       padding_value=self.tokenizer.pad_token_id)
        batch["condition_attention_mask"] = torch.nn.utils.rnn.pad_sequence(
            [torch.ones_like(c, dtype=torch.bool) for c in cond], batch_first=True, padding_value=False)
        return batch
