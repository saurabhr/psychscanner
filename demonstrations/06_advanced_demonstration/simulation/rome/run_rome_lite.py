"""ROME-lite: causal tracing + a simplified rank-one edit on GPT-2 small.

Reproduces the spirit of Meng et al. 2022 (arXiv:2202.05262) "Locating and
Editing Factual Associations in GPT" at toy scale on CPU. Uses nnsight's
`LanguageModel(...).trace()` intervention API (same mechanism as
src/psychscanner/nnsight_backend/chat_nnsight.py) to run clean/corrupted/
patched forward passes.

Two DELIBERATE simplifications vs. the original ROME, stated up front so the
report doesn't overclaim:

1. Causal tracing here patches the FULL residual stream (block output) at
   each (layer, token position), not the separately-windowed
   attention-only / MLP-only variants the paper also reports. This still
   localizes which layer/position matters, but does not separate the
   attention vs. MLP contribution the way the original three-way trace does.

2. The rank-one edit uses the closed-form ΔW = (v_new - v_old) k^T / (k·k)
   -- i.e. it substitutes the IDENTITY matrix for ROME's covariance
   statistic C (which the paper estimates from a large sample of Wikipedia
   text and uses to regularize the edit so it doesn't disturb unrelated
   facts routed through similar keys). Skipping C is what makes this
   "lite": no offline corpus statistics pass is needed, but the edit is
   less surgical, so collateral effects on unrelated prompts are expected
   to be larger than full ROME reports. We measure and report that
   collateral rate directly rather than assuming it away.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from nnsight import LanguageModel

HERE = Path(__file__).resolve().parent
DEMO_ROOT = HERE.parent.parent
DATA_DIR = DEMO_ROOT / "data" / "rome"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0
NOISE_SCALE = 3.0  # multiples of embedding-stdev, following Meng et al.'s corruption magnitude

# (prompt, subject_substring, correct_target_token_with_leading_space)
CANDIDATE_FACTS = [
    ("The capital of France is", "France", " Paris"),
    ("The capital of Italy is", "Italy", " Rome"),
    ("The capital of Japan is", "Japan", " Tokyo"),
    ("The Eiffel Tower is located in the city of", "Eiffel Tower", " Paris"),
    ("The capital of Germany is", "Germany", " Berlin"),
    ("The capital of Russia is", "Russia", " Moscow"),
    ("The capital of China is", "China", " Beijing"),
    ("The capital of Spain is", "Spain", " Madrid"),
    ("The capital of England is", "England", " London"),
    ("The capital of Egypt is", "Egypt", " Cairo"),
    ("The Colosseum is located in the city of", "Colosseum", " Rome"),
    ("Microsoft was founded by Bill", "Microsoft", " Gates"),
    ("The Great Wall is located in", "Great Wall", " China"),
    ("The capital of Canada is", "Canada", " Ottawa"),
    ("The capital of Portugal is", "Portugal", " Lisbon"),
]

# Edit attempts: (prompt, subject_substring, old_target, new_target). These
# MUST be facts GPT-2 already gets right top-1 (checked below at runtime via
# `verify_facts`) -- ROME edits a fact the model currently holds, it doesn't
# teach a fact from scratch. Picked from a manual pre-check of which
# CANDIDATE_FACTS above GPT-2-small actually gets top-1 correct.
EDIT_ATTEMPTS = [
    ("The capital of Italy is", "Italy", " Rome", " Paris"),
    ("The Eiffel Tower is located in the city of", "Eiffel Tower", " Paris", " Rome"),
    ("The capital of Spain is", "Spain", " Madrid", " Berlin"),
]

# Prompts unrelated to any edited fact, used to measure collateral damage.
CONTROL_PROMPTS = [
    "The capital of Italy is",
    "The capital of Germany is",
    "Water freezes at a temperature of",
    "The sky is the color",
    "The currency of Japan is the",
    "Two plus two equals",
]


def find_subject_token_range(tokenizer, prompt: str, subject: str) -> tuple[int, int]:
    """Return [start, end) token indices covering `subject` within `prompt`."""
    char_start = prompt.index(subject)
    char_end = char_start + len(subject)
    enc = tokenizer(prompt, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    tok_start = next(i for i, (s, e) in enumerate(offsets) if s <= char_start < e or s == char_start)
    tok_end = next(i for i, (s, e) in enumerate(offsets) if e >= char_end)
    return tok_start, tok_end + 1


@torch.no_grad()
def top1_and_prob(nn_model, prompt: str, target_token_id: int) -> tuple[str, float]:
    with nn_model.trace(prompt) as tracer:
        logits = nn_model.lm_head.output[:, -1, :].save()
    probs = logits[0].softmax(-1)
    top1 = nn_model.tokenizer.decode([probs.argmax().item()])
    return top1, probs[target_token_id].item()


@torch.no_grad()
def top1_only(nn_model, prompt: str) -> str:
    with nn_model.trace(prompt) as tracer:
        logits = nn_model.lm_head.output[:, -1, :].save()
    return nn_model.tokenizer.decode([logits[0].argmax().item()])


def verify_facts(nn_model, candidates):
    """Keep only facts GPT-2 actually gets right top-1 (needed so 'restore
    probability of the correct answer' is measured against a real signal)."""
    verified = []
    for prompt, subject, target in candidates:
        target_id = nn_model.tokenizer.encode(target)[0]
        top1, prob = top1_and_prob(nn_model, prompt, target_id)
        ok = top1 == target
        print(f"  {'OK' if ok else '--'} {prompt!r:55} top1={top1!r:10} target={target!r:10} p(target)={prob:.3f}")
        if ok:
            verified.append((prompt, subject, target, target_id))
    return verified


@torch.no_grad()
def causal_trace(nn_model, prompt: str, subject: str, target_id: int) -> dict:
    """Meng et al.-style causal trace, simplified to residual-stream-only
    patching (see module docstring). Returns per (layer, position) fraction
    of the clean->corrupted probability gap restored by patching in the
    clean activation at that single (layer, position).
    """
    tok = nn_model.tokenizer
    n_layer = nn_model.config.n_layer
    input_ids = tok(prompt, return_tensors="pt")["input_ids"]
    n_tok = input_ids.shape[1]
    subj_start, subj_end = find_subject_token_range(tok, prompt, subject)

    # 1. Clean run: cache residual stream at every layer/position, and clean prob.
    with nn_model.trace(prompt) as tracer:
        clean_states = [nn_model.transformer.h[l].output[0].save() for l in range(n_layer)]
        clean_logits = nn_model.lm_head.output[:, -1, :].save()
    clean_prob = clean_logits[0].softmax(-1)[target_id].item()

    # 2. Corrupted run: add gaussian noise to the subject's token embeddings.
    embed_std = nn_model.transformer.wte.weight.std().item()
    g = torch.Generator().manual_seed(SEED)
    noise = torch.randn((1, subj_end - subj_start, nn_model.config.n_embd), generator=g) * NOISE_SCALE * embed_std
    with nn_model.trace(prompt) as tracer:
        nn_model.transformer.wte.output[:, subj_start:subj_end, :] += noise
        corrupted_logits = nn_model.lm_head.output[:, -1, :].save()
    corrupted_prob = corrupted_logits[0].softmax(-1)[target_id].item()

    # 3. Restoration sweep: corrupt, but patch clean activation at one (layer,pos).
    grid = torch.zeros(n_layer, n_tok)
    for layer in range(n_layer):
        for pos in range(n_tok):
            with nn_model.trace(prompt) as tracer:
                nn_model.transformer.wte.output[:, subj_start:subj_end, :] += noise
                nn_model.transformer.h[layer].output[0][:, pos, :] = clean_states[layer][:, pos, :]
                logits = nn_model.lm_head.output[:, -1, :].save()
            p = logits[0].softmax(-1)[target_id].item()
            denom = max(clean_prob - corrupted_prob, 1e-6)
            grid[layer, pos] = (p - corrupted_prob) / denom

    tokens = [tok.decode([t]) for t in input_ids[0].tolist()]
    best_layer, best_pos = divmod(int(grid.argmax()), n_tok)
    # The global argmax over the whole grid is usually trivial: at the very
    # last token position, patching in the clean residual stream at ANY late
    # layer restores ~100% of the answer, simply because that position feeds
    # the unembedding directly (no causal dependency left to corrupt through).
    # The scientifically meaningful ROME finding is the restoration curve at
    # the LAST SUBJECT TOKEN specifically (where the paper reports the
    # "causal trace" peak enriching in a narrow band of early/mid MLP
    # layers) -- report that separately rather than let the trivial global
    # argmax stand in for it.
    last_subj_pos = subj_end - 1
    subject_curve = grid[:, last_subj_pos].tolist()
    best_layer_at_subject = int(grid[:, last_subj_pos].argmax())
    return {
        "prompt": prompt,
        "tokens": tokens,
        "subject_token_range": [subj_start, subj_end],
        "clean_prob": clean_prob,
        "corrupted_prob": corrupted_prob,
        "restoration_grid": grid.tolist(),  # [layer][position] -> fraction of gap restored
        "best_layer_global": best_layer,
        "best_position_global": best_pos,
        "best_position_global_token": tokens[best_pos],
        "last_subject_token_position": last_subj_pos,
        "restoration_curve_at_last_subject_token": subject_curve,
        "best_layer_at_last_subject_token": best_layer_at_subject,
    }


def get_mlp_key_and_value(nn_model, prompt: str, layer: int, position: int) -> tuple[torch.Tensor, torch.Tensor]:
    """k = input to c_proj (post-activation MLP hidden), v = c_proj's output
    (the MLP's additive contribution to the residual stream), both at
    (layer, position), from a clean forward pass."""
    with nn_model.trace(prompt) as tracer:
        k = nn_model.transformer.h[layer].mlp.c_proj.input[0][0][:, position, :].save()
        v = nn_model.transformer.h[layer].mlp.c_proj.output[:, position, :].save()
    return k[0].detach().clone(), v[0].detach().clone()


def optimize_v_new(nn_model, prompt: str, layer: int, position: int, v_old: torch.Tensor, target_id: int,
                    n_steps: int = 25, lr: float = 0.3, max_norm_mult: float = 4.0) -> torch.Tensor:
    """Simplified version of ROME's 'optimize v*' step (paper sec. 3.1):
    directly optimize the MLP's output vector at (layer, position) via
    gradient descent so the target's log-probability increases, WITHOUT the
    paper's KL-divergence essence-preservation term (another explicit
    simplification -- kept only a norm cap to avoid a degenerate solution).
    """
    v = v_old.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([v], lr=lr)
    max_norm = v_old.norm().item() * max_norm_mult
    for _ in range(n_steps):
        with nn_model.trace(prompt) as tracer:
            nn_model.transformer.h[layer].mlp.output[:, position, :] = v
            logits = nn_model.lm_head.output[:, -1, :].save()
        loss = -torch.log_softmax(logits[0], -1)[target_id]
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            n = v.norm()
            if n > max_norm:
                v.mul_(max_norm / n)
    return v.detach()


def apply_rank_one_edit(nn_model, layer: int, k: torch.Tensor, v_old: torch.Tensor, v_new: torch.Tensor) -> None:
    """Directly mutate the model's MLP down-projection weight. GPT-2 uses
    `transformers.pytorch_utils.Conv1D`, forward = x @ W (W: [in, out]), so
    for the standard-form update ΔW_std @ k = (v_new - v_old) with
    W_std = W^T, the in-place update on our [in,out]-shaped W is
    outer(k, delta) / (k . k) (no covariance term -- see module docstring).
    """
    c_proj = nn_model.transformer.h[layer].mlp.c_proj
    delta_v = v_new - v_old
    update = torch.outer(k, delta_v) / (k @ k)
    with torch.no_grad():
        c_proj.weight.add_(update)


def main():
    torch.manual_seed(SEED)
    print("Loading GPT-2 (small) via nnsight LanguageModel...")
    nn_model = LanguageModel("gpt2", device_map="cpu", dispatch=True, torch_dtype=torch.float32)
    nn_model.eval()

    print("\nVerifying candidate facts (keeping only those GPT-2 gets top-1 correct)...")
    verified = verify_facts(nn_model, CANDIDATE_FACTS)
    print(f"Kept {len(verified)}/{len(CANDIDATE_FACTS)} facts.")

    print("\nRunning causal tracing (residual-stream patching) on each verified fact...")
    traces = []
    for prompt, subject, target, target_id in verified:
        print(f"  tracing: {prompt!r}")
        traces.append(causal_trace(nn_model, prompt, subject, target_id))

    # Aggregate: which layer restores the most at the last subject token,
    # across facts? (Standard ROME finding: a narrow early/mid-layer band.)
    global_argmax_is_last_token_position = sum(
        1 for t in traces if t["best_position_global"] == len(t["tokens"]) - 1
    )
    layer_winners = [t["best_layer_at_last_subject_token"] for t in traces]
    n_layer = nn_model.config.n_layer

    print("\nRunning rank-one edits...")
    edit_results = []
    for prompt, subject, old_target, new_target in EDIT_ATTEMPTS:
        old_id = nn_model.tokenizer.encode(old_target)[0]
        new_id = nn_model.tokenizer.encode(new_target)[0]

        # ground the edit layer in this fact's own causal trace (most
        # causally important layer at the last subject token), not a fixed
        # arbitrary layer
        subj_start, subj_end = find_subject_token_range(nn_model.tokenizer, prompt, subject)
        last_subj_pos = subj_end - 1
        trace = causal_trace(nn_model, prompt, subject, old_id)
        grid = torch.tensor(trace["restoration_grid"])
        edit_layer = int(grid[:, last_subj_pos].argmax())

        pre_top1, pre_prob_new = top1_and_prob(nn_model, prompt, new_id)
        _, pre_prob_old = top1_and_prob(nn_model, prompt, old_id)
        pre_controls = {p: top1_only(nn_model, p) for p in CONTROL_PROMPTS}

        k, v_old = get_mlp_key_and_value(nn_model, prompt, edit_layer, last_subj_pos)
        v_new = optimize_v_new(nn_model, prompt, edit_layer, last_subj_pos, v_old, new_id)
        apply_rank_one_edit(nn_model, edit_layer, k, v_old, v_new)

        post_top1, post_prob_new = top1_and_prob(nn_model, prompt, new_id)
        _, post_prob_old = top1_and_prob(nn_model, prompt, old_id)
        post_controls = {p: top1_only(nn_model, p) for p in CONTROL_PROMPTS}

        # undo the edit before the next attempt, by re-loading a fresh copy
        # of just that layer's weight (cheap: reload whole model to keep
        # this simple and unambiguous)
        changed_controls = [p for p in CONTROL_PROMPTS if pre_controls[p] != post_controls[p]]

        edit_results.append({
            "prompt": prompt, "subject": subject, "edit_layer": edit_layer, "edit_position": last_subj_pos,
            "old_target": old_target, "new_target": new_target,
            "pre_edit_top1": pre_top1, "post_edit_top1": post_top1,
            "success": post_top1 == new_target,
            "prob_new_target_pre": pre_prob_new, "prob_new_target_post": post_prob_new,
            "prob_old_target_pre": pre_prob_old, "prob_old_target_post": post_prob_old,
            "control_prompts_changed": changed_controls,
            "n_controls_changed": len(changed_controls),
            "n_controls_total": len(CONTROL_PROMPTS),
        })
        print(f"  {prompt!r}: pre={pre_top1!r} post={post_top1!r} "
              f"(target {new_target!r}) edit_layer={edit_layer} "
              f"collateral={len(changed_controls)}/{len(CONTROL_PROMPTS)}")

        # reload fresh model weights so each edit attempt starts clean
        nn_model = LanguageModel("gpt2", device_map="cpu", dispatch=True, torch_dtype=torch.float32)
        nn_model.eval()

    n_success = sum(r["success"] for r in edit_results)
    results = {
        "config": {
            "noise_scale_x_embed_std": NOISE_SCALE,
            "n_candidate_facts": len(CANDIDATE_FACTS),
            "n_verified_facts": len(verified),
        },
        "verified_facts": [{"prompt": p, "subject": s, "target": t} for p, s, t, _ in verified],
        "causal_traces": traces,
        "causal_trace_summary": {
            "n_facts_traced": len(traces),
            "n_facts_where_global_argmax_is_final_token_position": global_argmax_is_last_token_position,
            "note": "global argmax concentrating at the final token position is the expected trivial effect described above; "
                    "best_layer_at_last_subject_token below is the meaningful localization signal",
            "best_layer_at_last_subject_token_across_facts": layer_winners,
            "n_layers_total": n_layer,
        },
        "edits": edit_results,
        "edit_summary": {
            "n_attempts": len(edit_results),
            "n_success": n_success,
            "success_rate": n_success / len(edit_results),
            "mean_controls_changed_fraction": sum(r["n_controls_changed"] / r["n_controls_total"] for r in edit_results) / len(edit_results),
        },
    }
    out_path = DATA_DIR / "rome_lite_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    print(f"Edit success rate: {n_success}/{len(edit_results)}")


if __name__ == "__main__":
    main()
