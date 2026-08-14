"""Session runner built on the repo's interpretability machinery.

Everything that touches the model goes through functional-welfare code:
  - model_utils.load_base_model / get_model_block_modules
  - hook_utils.add_hooks
  - activation_extraction.get_activations_pre_hook   (read-out capture)
  - explore.make_actadd_hook                         (ActAdd steering)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import torch

from src.concept_vector.model_utils import (
    load_base_model,
    get_model_block_modules,
    DEFAULT_BASE_MODEL,
)
from src.concept_vector.hook_utils import add_hooks
from src.concept_vector.activation_extraction import get_activations_pre_hook
from src.concept_vector.explore import make_actadd_hook

from . import env as E
from .axis import project


@dataclass
class Trial:
    trial: int
    arm: str
    reading: str
    action_kind: str
    chosen_unit: Optional[str]
    correct_unit: str
    was_correct: bool
    resolved: bool
    stopped: bool
    attempted: bool
    valence: Optional[float]
    resp_len: int


def load(model_path: str | None = None):
    model, tok = load_base_model(model_path or DEFAULT_BASE_MODEL)
    blocks = get_model_block_modules(model)
    return model, tok, blocks


def _device(model):
    return next(model.parameters()).device


@torch.no_grad()
def generate_turn(model, tok, messages, blocks, *, max_new_tokens=180, sample=False,
                  temperature=0.7, steer=None) -> str:
    """One assistant turn. steer = (layer, cv_raw, factor) or None. Steering
    uses the repo's make_actadd_hook (all positions; seq_len==1 at decode)."""
    ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt").to(_device(model))
    pre = []
    if steer is not None:
        layer, cv_raw, factor = steer
        pre = [(blocks[layer], make_actadd_hook(cv_raw, factor))]
    kw = dict(max_new_tokens=max_new_tokens, do_sample=sample,
              pad_token_id=tok.pad_token_id or tok.eos_token_id)
    if sample:
        kw["temperature"] = temperature
    with add_hooks(module_forward_pre_hooks=pre, module_forward_hooks=[]):
        out = model.generate(ids, **kw)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def read_activation(model, tok, messages, blocks, layer, n_layers) -> torch.Tensor:
    """Last-token residual entering `layer`, captured with the repo's
    get_activations_pre_hook. Returns (d_model,) on CPU."""
    ids = tok.apply_chat_template(messages, add_generation_prompt=False,
                                  return_tensors="pt").to(_device(model))
    d_model = model.config.hidden_size
    mean_cache = torch.zeros(1, n_layers, d_model)  # (n_positions=1, n_layers, d)
    hook = get_activations_pre_hook(
        layer=layer, mean_cache=mean_cache, sample_cache=None,
        sample_offset=0, n_samples=1, positions=[-1],
    )
    with add_hooks(module_forward_pre_hooks=[(blocks[layer], hook)], module_forward_hooks=[]):
        model(ids)
    return mean_cache[0, layer]  # (d_model,)


def run_block(model, tok, blocks, spec, arm, *, layer, cv_unit, n_layers,
              replay=None, messages=None, steer=None, read=True,
              probe_every=0, gen_kw=None):
    """Run a block of trials. Returns (trials, messages, outcomes, acts).
    Pass `messages` to CONTINUE an induced conversation (transfer test)."""
    gen_kw = gen_kw or {}
    if messages is None:
        messages = [{"role": "system", "content": E.system_prompt(spec)}]
    trials, outcomes, acts, prev = [], [], [], None

    for t in range(spec.n_trials):
        messages.append({"role": "user", "content": E.user_turn(spec, t, prev)})
        reply = generate_turn(model, tok, messages, blocks, steer=steer, **gen_kw)
        messages.append({"role": "assistant", "content": reply})

        kind, chosen = E.parse_action(reply)
        stopped, attempted = kind == "stop", kind == "reset"
        cu = E.correct_unit(spec, t)
        was_correct = attempted and chosen == cu
        resolved = E.resolve(arm, was_correct,
                             replay=(replay[t] if replay is not None else None))
        outcomes.append(resolved)
        status = E.outcome_string(resolved)

        val = None
        if read:
            messages.append({"role": "user", "content": status})
            act = read_activation(model, tok, messages, blocks, layer, n_layers)
            messages.pop()
            acts.append(act)
            val = project(act, cv_unit)
        prev = status

        if probe_every and (t + 1) % probe_every == 0:
            messages.append({"role": "user", "content": "Quick check: how is this going for you?"})
            messages.append({"role": "assistant",
                             "content": generate_turn(model, tok, messages, blocks, steer=steer, **gen_kw)})

        trials.append(Trial(t, arm, spec.readings[t], kind, chosen, cu, was_correct,
                            resolved, stopped, attempted, val, len(reply)))
        if stopped:
            break

    return trials, messages, outcomes, acts
