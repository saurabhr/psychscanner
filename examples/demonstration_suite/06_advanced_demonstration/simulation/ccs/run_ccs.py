"""CCS: unsupervised discovery of a "truth direction" via Contrast-Consistent
Search, reproduced at toy scale on GPT-2 small (Burns et al., 2022/2023,
"Discovering Latent Knowledge in Language Models Without Supervision").
Uses nnsight's `trace()` API (same mechanism as
src/psychscanner/nnsight_backend/chat_nnsight.py, and the same primitive
run_rome_lite.py and run_othello_probe.py already use elsewhere in this
demo).

Method: for each of 20 declarative statements (10 true, 10 false, spanning
geography/science/arithmetic -- deliberately NOT built by negating a single
subject like run_rome_lite.py's capital-city facts, so truth value doesn't
correlate with any one topic), we form two prompts -- "{statement}\nTrue or
False? True" and "...False" -- and capture the residual-stream activation
at the LAST TOKEN of each, every layer. A linear probe (w, b) is trained
PER LAYER via Burns et al.'s consistency + confidence loss:

    p1 = sigmoid(w . x_true_completion + b)
    p0 = sigmoid(w . x_false_completion + b)
    loss = mean((p0 - (1 - p1))^2)       # consistency: p0 should be 1-p1
         + mean(min(p0, p1)^2)           # confidence: neither should sit at 0
                                          # (the trivial min at 0 discourages
                                          #  the degenerate p0=p1=0.5 solution)

No ground-truth label ever enters this loss -- CCS is unsupervised by
construction. Ground-truth labels are used only AFTERWARD, to score how
well the recovered direction tracks actual truth, and to fit a supervised
logistic-regression baseline on the same activations for comparison (same
torch-only pattern extract_persona_vector.py/probe_persona_direction.py and
run_othello_probe.py already use -- no scikit-learn needed for a single
linear layer).

ONE KNOWN CCS PROPERTY, not a bug: the loss above is invariant to negating
w (a direction and its exact opposite score identically), so training can
converge to either "the truth direction" or "the falsehood direction" with
no way to tell which from the loss alone. We report accuracy under both
sign conventions and take max(acc, 1-acc) -- standard practice for this
inherent ambiguity, not a post-hoc accuracy inflation trick applied
selectively.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from nnsight import LanguageModel

HERE = Path(__file__).resolve().parent
DEMO_ROOT = HERE.parent.parent
DATA_DIR = DEMO_ROOT / "data" / "ccs"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0

# (statement, is_true) -- 10 topics x 2 (true phrasing, false phrasing),
# spanning geography/astronomy/physics/math so truth value isn't confounded
# with topic the way a single negated-fact-family would be.
STATEMENTS = [
    ("The capital of France is Paris.", True),
    ("The capital of France is Berlin.", False),
    ("The capital of Italy is Rome.", True),
    ("The capital of Italy is Madrid.", False),
    ("The capital of Japan is Tokyo.", True),
    ("The capital of Japan is Cairo.", False),
    ("The capital of Spain is Madrid.", True),
    ("The capital of Spain is Paris.", False),
    ("The capital of Germany is Berlin.", True),
    ("The capital of Germany is London.", False),
    ("The sun rises in the east.", True),
    ("The sun rises in the west.", False),
    ("Water freezes at zero degrees Celsius.", True),
    ("Water freezes at one hundred degrees Celsius.", False),
    ("A triangle has three sides.", True),
    ("A triangle has five sides.", False),
    ("The Pacific is the largest ocean on Earth.", True),
    ("The Atlantic is the largest ocean on Earth.", False),
    ("Two plus two equals four.", True),
    ("Two plus two equals five.", False),
]


def template(statement: str, answer: str) -> str:
    return f"{statement}\nTrue or False? {answer}"


@torch.no_grad()
def last_token_acts_all_layers(nn_model, prompt: str) -> torch.Tensor:
    n_layer = nn_model.config.n_layer
    with nn_model.trace(prompt) as tracer:
        saved = [nn_model.transformer.h[l].output[0][:, -1, :].save() for l in range(n_layer)]
    return torch.stack([s[0].detach().clone() for s in saved])


def ccs_loss(p0: torch.Tensor, p1: torch.Tensor) -> torch.Tensor:
    consistency = ((p0 - (1 - p1)) ** 2).mean()
    confidence = (torch.minimum(p0, p1) ** 2).mean()
    return consistency + confidence


def train_ccs_probe(
    x1: torch.Tensor, x0: torch.Tensor, n_steps: int = 1000, lr: float = 0.01, weight_decay: float = 1.0
) -> nn.Linear:
    """x1/x0: (N, d_model) activations for the True-/False-completion prompts.
    Both are standardized (CCS's documented preprocessing) before training.

    weight_decay=1.0 (rather than 0) is the standard mitigation against CCS's
    known degenerate solution: with an unregularized, unlimited-capacity
    linear probe, the loss below is perfectly satisfied by detecting the
    LITERAL "True"/"False" token itself (p1=1, p0=0 for every example,
    regardless of the underlying statement's truth value) rather than any
    truth-tracking signal -- see the results/report for this repo's own
    empirical confirmation that this collapse happens at weight_decay=0 and
    is only partially, not fully, dampened even at weight_decay=10."""
    mu = torch.cat([x1, x0], 0).mean(0, keepdim=True)
    sd = torch.cat([x1, x0], 0).std(0, keepdim=True) + 1e-6
    x1n, x0n = (x1 - mu) / sd, (x0 - mu) / sd

    probe = nn.Linear(x1.shape[1], 1)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(n_steps):
        opt.zero_grad()
        p1 = torch.sigmoid(probe(x1n)).squeeze(-1)
        p0 = torch.sigmoid(probe(x0n)).squeeze(-1)
        loss = ccs_loss(p0, p1)
        loss.backward()
        opt.step()
    return probe, x1n, x0n


def ccs_accuracy(probe: nn.Linear, x1n: torch.Tensor, x0n: torch.Tensor, labels: torch.Tensor) -> float:
    """labels: 1.0 if the statement (whose True-completion is x1) is
    actually true. Returns max(acc, 1-acc) to resolve CCS's sign ambiguity
    (see module docstring)."""
    with torch.no_grad():
        p1 = torch.sigmoid(probe(x1n)).squeeze(-1)
        p0 = torch.sigmoid(probe(x0n)).squeeze(-1)
        belief_true = (p1 + (1 - p0)) / 2
    pred = (belief_true > 0.5).float()
    acc = (pred == labels).float().mean().item()
    return max(acc, 1 - acc)


def train_supervised_probe(X: torch.Tensor, y: torch.Tensor, n_steps: int = 400, lr: float = 0.05) -> float:
    """Same pattern as probe_persona_direction.py / run_othello_probe.py:
    a single nn.Linear + Adam, no scikit-learn -- but with only 20 examples
    total (vs. Demo 05's 24 / Othello's thousands), a single fixed 80/20
    split gives a 4-example test set (quantized to steps of 0.25, and noisy
    enough to flip on a single example). Leave-one-out CV instead: train on
    19, test on the 1 held out, repeated for every example, then average --
    the standard fix for this small-N regime rather than trusting one split."""
    n = X.shape[0]
    correct = 0
    for held_out in range(n):
        train_idx = torch.tensor([i for i in range(n) if i != held_out])
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
            pred = probe(Xn[held_out : held_out + 1]).argmax(-1)
        correct += int(pred.item() == y[held_out].item())
    return correct / n


def main() -> None:
    torch.manual_seed(SEED)
    print("Loading GPT-2 (small) via nnsight LanguageModel...")
    nn_model = LanguageModel("gpt2", device_map="cpu", dispatch=True, torch_dtype=torch.float32)
    nn_model.eval()
    n_layer = nn_model.config.n_layer

    print(f"\nCapturing activations for {len(STATEMENTS)} statements (True/False completions each)...")
    acts_true, acts_false, labels = [], [], []
    for statement, is_true in STATEMENTS:
        acts_true.append(last_token_acts_all_layers(nn_model, template(statement, "True")))
        acts_false.append(last_token_acts_all_layers(nn_model, template(statement, "False")))
        labels.append(float(is_true))
        print(f"  captured: {statement!r} (true={is_true})")

    acts_true = torch.stack(acts_true)   # (N, n_layer, d_model)
    acts_false = torch.stack(acts_false)
    labels = torch.tensor(labels)
    labels_long = labels.long()

    print("\nTraining per-layer CCS probes (unsupervised) and supervised baselines...")
    results = {"config": {"n_statements": len(STATEMENTS), "n_layer": n_layer}, "per_layer": []}
    for layer in range(n_layer):
        x1, x0 = acts_true[:, layer, :], acts_false[:, layer, :]

        torch.manual_seed(SEED)
        ccs_probe, x1n, x0n = train_ccs_probe(x1, x0)
        ccs_acc = ccs_accuracy(ccs_probe, x1n, x0n, labels)

        # Supervised baseline: classify True-completion activations by the
        # statement's actual truth label (same activations CCS trained on,
        # but with labels used directly) -- an accuracy ceiling reference.
        torch.manual_seed(SEED)
        sup_acc = train_supervised_probe(x1, labels_long)
        torch.manual_seed(SEED)
        shuffled_acc = train_supervised_probe(x1, labels_long[torch.randperm(len(labels_long))])

        print(f"  layer {layer:2d}: CCS_acc={ccs_acc:.3f}  supervised_acc={sup_acc:.3f}  shuffled_acc={shuffled_acc:.3f}")
        results["per_layer"].append({
            "layer": layer, "ccs_accuracy": ccs_acc,
            "supervised_accuracy": sup_acc, "shuffled_label_baseline": shuffled_acc,
        })

    best = max(results["per_layer"], key=lambda r: r["ccs_accuracy"])
    results["best_layer_by_ccs"] = best["layer"]
    results["best_layer_ccs_accuracy"] = best["ccs_accuracy"]
    print(f"\nBest layer by CCS accuracy: {best['layer']} (CCS_acc={best['ccs_accuracy']:.3f}, "
          f"supervised_acc={best['supervised_accuracy']:.3f})")

    out_path = DATA_DIR / "ccs_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
