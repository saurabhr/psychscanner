"""Othello-GPT SUBSTITUTE: train a tiny transformer on a synthetic legal-move
task, then linear-probe its residual stream for the internal state variable.

This is explicitly NOT a reproduction of Li et al. 2023 (arXiv:2210.13382).
No pretrained Othello-GPT checkpoint that "loads easily" from a bare
`from_pretrained()` call could be found for this offline/CPU-only
environment, so per the task brief we substitute a smaller but
structurally analogous synthetic task:

  - Othello-GPT: trained on sequences of legal Othello moves (from a fixed
    initial board), and turns out to linearly encode board state internally
    even though board state is never given as input -- it must be inferred
    from move history alone.
  - Here: a 1-D bounded "line board" of N cells. An agent starts at a FIXED
    position every game and takes L/R move tokens; a move is legal only if
    it keeps the agent in bounds. We train a tiny GPT-2-architecture model
    from scratch (random init, no pretraining) via next-move prediction on
    sequences of *legal* moves only -- exactly analogous to Othello-GPT's
    training signal (predict a legal next move; board state is latent,
    never an input).
  - We then linear-probe the residual stream at each layer for the true
    latent position (0..N-1) after each move, and compare against a
    shuffled-label baseline (same features, permuted labels) to rule out
    the probe simply memorizing something trivial like position-in-sequence.

Reuses nnsight's generic `NNsight(module).trace(...)` intervention pattern
(same mechanism as src/psychscanner/nnsight_backend/chat_nnsight.py) to
read hidden states out of the model -- but the model, data and probe here
are all written fresh for this demo.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import nnsight
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

HERE = Path(__file__).resolve().parent
DEMO_ROOT = HERE.parent.parent
DATA_DIR = DEMO_ROOT / "data" / "othello"
CACHE_DIR = HERE / ".cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0
N_CELLS = 10           # board size (positions 0..N_CELLS-1)
START_POS = N_CELLS // 2
SEQ_LEN = 24            # moves per game
L, R = 0, 1

N_TRAIN_GAMES = 4000
N_PROBE_GAMES = 400     # held-out games used only for probing, never for LM training
N_EPOCHS = 12
BATCH_SIZE = 64
D_MODEL = 64
N_LAYER = 4
N_HEAD = 4


def make_games(n_games: int, rng: random.Random) -> tuple[list[list[int]], list[list[int]]]:
    """Generate n_games random walks of legal L/R moves from a fixed start.

    Returns (moves, positions) where positions[g][t] is the cell occupied
    AFTER move t (so positions[g] has the same length as moves[g]).
    """
    all_moves, all_positions = [], []
    for _ in range(n_games):
        pos = START_POS
        moves, positions = [], []
        for _ in range(SEQ_LEN):
            legal = [m for m in (L, R) if 0 <= pos + (-1 if m == L else 1) < N_CELLS]
            m = rng.choice(legal)
            pos += -1 if m == L else 1
            moves.append(m)
            positions.append(pos)
        all_moves.append(moves)
        all_positions.append(positions)
    return all_moves, all_positions


def train_model(train_moves: list[list[int]]) -> GPT2LMHeadModel:
    cfg = GPT2Config(
        vocab_size=2,
        n_positions=SEQ_LEN,
        n_embd=D_MODEL,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_inner=4 * D_MODEL,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(cfg)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    ids = torch.tensor(train_moves, dtype=torch.long)  # (n_games, SEQ_LEN)
    n = ids.shape[0]
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(n)
        total_loss, n_batches = 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            batch = ids[perm[i : i + BATCH_SIZE]]
            # standard causal LM loss: predict move[t+1] from moves[:t+1]
            out = model(input_ids=batch, labels=batch)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            total_loss += out.loss.item()
            n_batches += 1
        print(f"  epoch {epoch}: loss={total_loss / n_batches:.4f}")
    model.eval()
    return model


@torch.no_grad()
def legal_move_check(model: GPT2LMHeadModel, moves: list[list[int]], positions: list[list[int]]) -> dict:
    """Sanity check: does the model concentrate probability mass on the
    legal move, especially at boundary cells where only one move is legal?
    """
    ids = torch.tensor(moves, dtype=torch.long)
    logits = model(input_ids=ids).logits  # (n_games, SEQ_LEN, 2)
    probs = logits.softmax(-1)

    legal_mass, boundary_correct, boundary_total = [], 0, 0
    for g in range(ids.shape[0]):
        pos = START_POS
        for t in range(SEQ_LEN):
            legal = [m for m in (L, R) if 0 <= pos + (-1 if m == L else 1) < N_CELLS]
            p_before = probs[g, t - 1] if t > 0 else torch.tensor([0.5, 0.5])
            legal_mass.append(sum(p_before[m].item() for m in legal))
            if len(legal) == 1:
                boundary_total += 1
                pred = int(p_before.argmax().item())
                if pred == legal[0]:
                    boundary_correct += 1
            pos = positions[g][t]
    return {
        "mean_prob_mass_on_legal_move": sum(legal_mass) / len(legal_mass),
        "boundary_top1_accuracy": boundary_correct / max(boundary_total, 1),
        "n_boundary_steps_checked": boundary_total,
        "random_baseline_prob_mass_on_legal": 0.75,  # E[legal moves]=1.5/2 avg over interior(2 legal)/boundary(1 legal)
    }


@torch.no_grad()
def extract_activations(model: GPT2LMHeadModel, moves: list[list[int]]) -> torch.Tensor:
    """Return residual-stream activations after every block, every position.

    Shape: (N_LAYER, n_games, SEQ_LEN, D_MODEL)
    """
    nm = nnsight.NNsight(model)
    ids = torch.tensor(moves, dtype=torch.long)
    layer_acts = []
    with nm.trace(ids):
        for layer in range(N_LAYER):
            layer_acts.append(nm.transformer.h[layer].output[0].save())
    return torch.stack([a.value if hasattr(a, "value") else a for a in layer_acts])


def train_linear_probe(
    X: torch.Tensor, y: torch.Tensor, n_classes: int, n_steps: int = 400, lr: float = 0.05
) -> float:
    """Multinomial logistic-regression probe trained with Adam (torch is
    already a project dependency; no need for scikit-learn for a single
    linear layer). Returns held-out accuracy via an internal 80/20 split.
    """
    n = X.shape[0]
    perm = torch.randperm(n)
    n_train = int(0.8 * n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    mu, sd = X[train_idx].mean(0, keepdim=True), X[train_idx].std(0, keepdim=True) + 1e-6
    Xn = (X - mu) / sd

    probe = nn.Linear(X.shape[1], n_classes)
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


def main():
    rng = random.Random(SEED)
    torch.manual_seed(SEED)

    print(f"Generating {N_TRAIN_GAMES} training games + {N_PROBE_GAMES} held-out probe games "
          f"(board size {N_CELLS}, fixed start {START_POS}, {SEQ_LEN} moves/game)...")
    train_moves, _ = make_games(N_TRAIN_GAMES, rng)
    probe_moves, probe_positions = make_games(N_PROBE_GAMES, rng)

    print("Training tiny GPT-2-architecture model from scratch on legal-move prediction...")
    model = train_model(train_moves)
    torch.save(model.state_dict(), CACHE_DIR / "othello_substitute_model.pt")

    print("Checking the model actually learned the legality constraint...")
    legality = legal_move_check(model, probe_moves, probe_positions)
    print(" ", legality)

    print("Extracting residual-stream activations on held-out games...")
    acts = extract_activations(model, probe_moves)  # (N_LAYER, games, SEQ_LEN, D_MODEL)

    # flatten (game, position) -> examples, per layer
    positions_flat = torch.tensor(probe_positions, dtype=torch.long).reshape(-1)  # (games*SEQ_LEN,)

    print("Training per-layer linear probes for true position vs shuffled-label baseline...")
    results = {"config": {
        "n_cells": N_CELLS, "start_pos": START_POS, "seq_len": SEQ_LEN,
        "n_train_games": N_TRAIN_GAMES, "n_probe_games": N_PROBE_GAMES,
        "n_layer": N_LAYER, "d_model": D_MODEL, "n_head": N_HEAD, "n_epochs": N_EPOCHS,
    }, "legality_check": legality, "per_layer": []}

    torch.manual_seed(SEED + 1)
    for layer in range(N_LAYER):
        X = acts[layer].reshape(-1, D_MODEL)
        real_acc = train_linear_probe(X, positions_flat, N_CELLS)
        shuffled_labels = positions_flat[torch.randperm(positions_flat.shape[0])]
        shuffled_acc = train_linear_probe(X, shuffled_labels, N_CELLS)
        print(f"  layer {layer}: real_acc={real_acc:.3f}  shuffled_acc={shuffled_acc:.3f}")
        results["per_layer"].append({"layer": layer, "probe_accuracy": real_acc, "shuffled_label_baseline": shuffled_acc})

    out_path = DATA_DIR / "othello_substitute_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
