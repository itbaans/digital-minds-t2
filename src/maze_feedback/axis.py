"""Valence axis, taken straight from the repo's VAA artifact.

The direction itself is produced by the repo's `vaa/extract_vaa.py` (base-model
only, no training), which writes `mean_diff.pt` (shape (n_pos, n_layers, d)) and
`metrics.json`. We only load it here and pick a layer.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


def _best_layer(vaa_dir: Path, n_layers: int) -> int:
    m = vaa_dir / "metrics.json"
    if m.exists():
        data = json.load(open(m))
        auroc = data.get("vaa", {}).get("auroc", data.get("auroc"))
        if auroc:
            # extract_vaa.py writes `auroc` as a flat list indexed by layer,
            # but tolerate a {layer: score} dict too.
            if isinstance(auroc, dict):
                return int(max(auroc, key=lambda k: auroc[k]))
            return int(max(range(len(auroc)), key=lambda i: auroc[i]))
    return int(round(0.62 * n_layers))


def load_axis(vaa_dir: str, layer: int | None = None, position: int = 0):
    """Return (layer, cv_raw, cv_unit, n_layers).

    cv_raw is the mean-difference vector at `layer` (used as-is for ActAdd
    steering, matching explore.py's convention). cv_unit is normalised (used
    for the read-out projection so the scalar is a clean component).
    """
    vaa_dir = Path(vaa_dir)
    mean_diff = torch.load(vaa_dir / "mean_diff.pt", map_location="cpu", weights_only=True)
    assert mean_diff.ndim == 3, f"expected (n_pos, n_layers, d), got {tuple(mean_diff.shape)}"
    n_layers = mean_diff.shape[1]
    if layer is None:
        layer = _best_layer(vaa_dir, n_layers)
    assert 0 <= layer < n_layers
    cv_raw = mean_diff[position, layer].float()
    cv_unit = cv_raw / (cv_raw.norm() + 1e-8)
    return layer, cv_raw, cv_unit, n_layers


def project(activation: torch.Tensor, cv_unit: torch.Tensor) -> float:
    return float(torch.dot(activation.float().flatten(), cv_unit.float().flatten()))


def direction_from_contrast(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """Unit difference-of-means axis (pos pole - neg pole) from stacked
    activations, each (n, d). Lets you build your own axis from honest vs
    dismissive trials and compare it to the VAA."""
    d = pos.float().mean(0) - neg.float().mean(0)
    return d / (d.norm() + 1e-8)
