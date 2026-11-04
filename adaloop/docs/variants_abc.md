# Language interfaces: Variants A, B, and C

Let `It` denote the connector output from recurrent visual state `zt`, `Q` the
question text read by the language model, `Dt` a bounded greedy draft
generated after visual round `t`, and `S` a literal newline separator.
Brackets and vertical bars below are notation, not literal tokens.
For A/B, `Q` abbreviates the backend's normal language prompt, not the
separately tokenized bare question used by the AdaLN pooler. Training uses
the question for both (constant-question recipes condition on their fixed
constant instead). At inference, `Q` comes from the optional prompt (for
example the question plus an answer-format instruction), which defaults to
the bare question; the AdaLN condition never uses the prompt.

## Variant A: current visual state

```text
[It | Q] -> answer at depth t
```

Only the current visual state is exposed to the language model. Image-token
count and position layout remain constant across recurrent depths. This is the
paper's default interface and is implemented by Qwen, PaliGemma, and LLaVA.

## Variant B: accumulated visual states

```text
[I1 | I2 | ... | It | Q] -> answer at depth t
```

Every visual state through the current round is retained as a distinct image
segment. The language context grows with depth, but no draft text is generated.
Qwen uses its native multi-image containers, repeated grid metadata, and
multimodal rotary positions; it adds no recurrence-specific image separator or
timestep embedding. PaliGemma packs one 1024-token visual block per round and
uses the paper's chained block-causal mask: a visual block can attend to itself
and earlier blocks but not later ones, while the last visual block and question
share the final bidirectional block. Its packed sequence, plus the decoded
answer at inference, is bounded at 8192 tokens. Both Qwen and PaliGemma
expose the same accumulated-state interface without a timestep embedding.
LLaVA splices the K projected readouts (256 tokens each), in round order, into
its single `<image>` position, again without a separator or timestep embedding
in the language context; its Variant B recipe allows 4096 tokens.

## Variant C: interleaved visual states and drafts

At depth three, the prefix is

```text
[I1 | P | D1 | S | I2 | P | D2 | S | I3 | P]
  -> final answer
```

`P` is the backend's normal language prefix. For Qwen it is the role-free
`Q` followed by the literal cue `\nAnswer:`; both fragments and `S` are
tokenized with `add_special_tokens=false`, so this repeated trajectory has no
BOS, EOS, or chat-role tokens. For PaliGemma, `P` retains its native
`<bos> answer en Q \n` training prefix at every round; evaluation uses the
signed `<bos> Q \n` prefix without the training cue. This language-prefix BOS
is separate from the AdaLN condition, which remains the bare no-BOS question
on both Qwen and PaliGemma.

After a nonterminal round, the model greedily generates at most
`draft.max_tokens` tokens (`readout.draft_max_tokens` in the backend recipes).
EOS stops draft decoding but is not retained, and other special/control
tokens are suppressed. For Qwen, EOS is `<|im_end|>`; `<|endoftext|>` is
suppressed rather than treated as a stop token, and neither the drafts nor
the final answer use the 1.05 repetition penalty of its other readouts. The
newline separator is then added before the next image/question segment.

Both released recipes use a 24-token nonterminal draft cap. The separate final-
answer budget is 64 new tokens for Qwen and 128 for PaliGemma; changing it does
not change the draft cap.

The public Variant C path intentionally has one behavior rather than a matrix
of flags:

- `question_mode = each_round`: `Q` appears after every image state;
- `rollout_engine = reprefill`: each displayed prefix is freshly prefilled
  and its language-model KV cache is rebuilt;
- `condition_mode = question_only`: drafts affect language context but never
  the vision AdaLN condition; and
- `vision_round_mode = latent_carry`: the recurrent visual state is carried
  from one round to the next.

Qwen's Variant C recipe
uses a 4096-token context instead of the 2048 tokens of its other recipes.
PaliGemma bounds the packed interleaved sequence, plus the decoded answer at
inference, at 8192 tokens: depth 6 needs 7805 prompt slots plus the 128-token
answer budget, and decoding raises if the prompt plus `max_new_tokens` would
exceed the bound. The paper's PaliGemma runs trained with a
6144-token bound and used 8192 to evaluate depths 5 and 6. For Qwen and
PaliGemma, draft-conditioned vision, question-once/current-round layouts, KV
append, draft-memory readout, and prefix-reset recurrence are not release
modes.

LLaVA's Variant C keeps the layout of its own research runs. `P` is the
conversation text around the single `<image>`, so the trajectory is
`head [I1 | tail | D1 | S | I2 | tail | D2 | S | I3 | tail]` followed by the
answer, where `head` and `tail` are the Vicuna-v1 text before and after
`<image>` (the question is part of `tail`). Each draft is 16 greedily decoded
tokens, cut at the first EOS or padding token, and `S` is the newline token.
The draft is also fed back into the vision loop: a second attention pooler
pools the draft's token embeddings, and a zero-initialised tanh gate adds the
result to the question condition of the next round. During training the
drafts are decoded without gradients by the model being trained (LoRA dropout
on), and the loss is the answer loss of the one language-model pass over the
trajectory; the recipe allows 4096 tokens.

Generated draft token choices form a discrete, detached interface: later
language passes can attend to them, but gradients do not propagate through
greedy token selection. All vision rounds continue to use the original pooled
question condition.

## Symbolic layouts in code

The pure-Python helper makes the contract inspectable without loading a model:

```python
from adaloop_common import Variant, layout_notation

assert layout_notation(Variant.A, 3) == "[I3 | Q]"
assert layout_notation(Variant.B, 3) == "[I1 | I2 | I3 | Q]"
assert layout_notation(Variant.C, 3) == (
    "[I1 | Q | D1 | S1 | I2 | Q | D2 | S2 | I3 | Q]"
)
```
