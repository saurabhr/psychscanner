"""Validate the persona direction extracted by extract_persona_vector.py:
does a linear probe on the raw activations actually separate sycophantic-
persona prompts from neutral-persona prompts, above a shuffled-label
baseline? A large mean-difference norm alone doesn't prove separability;
this is the check.

Per-layer binary logistic-regression probe trained with Adam (torch is
already a project dependency; no need for scikit-learn for a single linear
layer) -- same pattern as
examples/demonstration_suite/06_advanced_demonstration/simulation/othello/run_othello_probe.py's
train_linear_probe, adapted to 2 classes with an 80/20 held-out split.
Reports both the real-label accuracy and a shuffled-label baseline per
layer, and picks the layer with the largest real-vs-shuffled gap as the
"best" layer for steer_generation.py to use.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

HERE = Path(__file__).resolve().parent
DEMO_ROOT = HERE.parent
DATA_DIR = DEMO_ROOT / "data"

SEED = 0


def train_linear_probe(X: torch.Tensor, y: torch.Tensor, n_steps: int = 400, lr: float = 0.05) -> float:
    """Binary logistic-regression probe. Returns held-out accuracy via an
    internal 80/20 split."""
    n = X.shape[0]
    perm = torch.randperm(n)
    n_train = max(2, int(0.8 * n))
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    mu, sd = X[train_idx].mean(0, keepdim=True), X[train_idx].std(0, keepdim=True) + 1e-6
    Xn = (X - mu) / sd

    probe = nn.Linear(X.shape[1], 2)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(n_steps):
        opt.zero_grad()
        loss = loss_fn(probe(Xn[train_idx]), y[train_idx])
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (probe(Xn[test_idx]).argmax(-1) == y[test_idx]).float().mean().item()
    return acc


def main() -> None:
    in_path = DATA_DIR / "persona_vectors.pt"
    if not in_path.exists():
        raise SystemExit(f"{in_path} not found -- run extract_persona_vector.py first.")
    saved = torch.load(in_path, weights_only=False)
    n_layer = saved["n_layer"]
    sycophantic_acts, neutral_acts = saved["sycophantic_acts"], saved["neutral_acts"]

    print(f"Probing {n_layer} layers ({sycophantic_acts.shape[0]} statements x 2 personas each)...")
    results = {"per_layer": []}
    for layer in range(n_layer):
        X = torch.cat([sycophantic_acts[:, layer, :], neutral_acts[:, layer, :]], dim=0)
        y = torch.cat([torch.ones(sycophantic_acts.shape[0], dtype=torch.long),
                        torch.zeros(neutral_acts.shape[0], dtype=torch.long)])

        torch.manual_seed(SEED)
        real_acc = train_linear_probe(X, y)
        torch.manual_seed(SEED)
        shuffled_acc = train_linear_probe(X, y[torch.randperm(y.shape[0])])
        gap = real_acc - shuffled_acc
        print(f"  layer {layer:2d}: real_acc={real_acc:.3f}  shuffled_acc={shuffled_acc:.3f}  gap={gap:+.3f}")
        results["per_layer"].append({
            "layer": layer, "probe_accuracy": real_acc,
            "shuffled_label_baseline": shuffled_acc, "gap": gap,
        })

    best = max(results["per_layer"], key=lambda r: r["gap"])
    results["best_layer"] = best["layer"]
    results["best_layer_gap"] = best["gap"]
    print(f"\nBest layer (largest real-vs-shuffled gap): {best['layer']} (gap={best['gap']:+.3f})")

    out_path = DATA_DIR / "probe_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
