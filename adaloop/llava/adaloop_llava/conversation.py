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
# llava/conversation.py: only the Vicuna-v1 template (separator style TWO) used by LLaVA-1.5 is
# kept, as its constants and a renderer for one user turn and its answer.

"""The Vicuna-v1 conversation template of LLaVA-1.5 prompts."""

from typing import Optional

SYSTEM = ("A chat between a curious user and an artificial intelligence assistant. "
          "The assistant gives helpful, detailed, and polite answers to the user's questions.")
USER, ASSISTANT = "USER", "ASSISTANT"
SEP, SEP2 = " ", "</s>"


def vicuna_v1_prompt(user: str, assistant: Optional[str] = None) -> str:
    """``"<system> USER: <user> ASSISTANT: <assistant></s>"``; with ``assistant`` None (or empty)
    the assistant turn is left open: ``"<system> USER: <user> ASSISTANT:"``."""
    ret = SYSTEM + SEP
    for role, message, sep in ((USER, user, SEP), (ASSISTANT, assistant, SEP2)):
        ret += f"{role}: {message}{sep}" if message else f"{role}:"
    return ret
