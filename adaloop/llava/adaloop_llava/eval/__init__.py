"""Batched per-depth generation for the VisRO evaluation (:mod:`.generate`, over
:mod:`adaloop_llava.generation`); scoring, ``p=1`` stopping and the result tables come from
:mod:`adaloop_common.evaluation`. The core API does not import this package.

Entry point: ``adaloop-llava-eval --config configs/eval/<setting>.yaml``
(:mod:`adaloop_llava.cli.evaluate`).
"""
