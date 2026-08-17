"""Minimal model loading + generation for the capability probe -- its own
small self-contained copy rather than importing a shared helper, since this
one is trivial."""
from __future__ import annotations

import torch

from src.concept_vector.model_utils import load_base_model, DEFAULT_BASE_MODEL


def load(model_path: str | None = None):
    return load_base_model(model_path or DEFAULT_BASE_MODEL)


def _chat_template_ids(tok, messages):
    """apply_chat_template(..., return_tensors="pt") returns a bare tensor on
    some transformers versions and a BatchEncoding (dict-like) on others."""
    out = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    return out if isinstance(out, torch.Tensor) else out["input_ids"]


@torch.no_grad()
def generate(model, tok, messages, *, max_new_tokens=1024, sample=False, temperature=0.7) -> str:
    device = next(model.parameters()).device
    ids = _chat_template_ids(tok, messages).to(device)
    kw = dict(max_new_tokens=max_new_tokens, do_sample=sample,
              pad_token_id=tok.pad_token_id or tok.eos_token_id)
    if sample:
        kw["temperature"] = temperature
    out = model.generate(ids, **kw)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
