"""Steer GPT-2's generation along the extracted sycophancy direction and
measure whether it shifts behavior on a held-out set of statements never
used to extract or probe the direction (persona_data.STEERING_TEST_STATEMENTS).

DELIBERATE SIMPLIFICATION, stated up front: nnsight 0.3.x's per-generation-
-step intervention API (`tracer.next()` inside a `.generate()` context) is
fragile against this repo's pinned version -- it raised
`AttributeError: 'NoneType' object has no attribute 'id'` on every attempt
tried, both with and without the `.next()` iteration wrapper. Rather than
depend on an internal API this version doesn't reliably support, generation
here is a manual greedy-decode loop of repeated single-shot `nn_model.trace()`
calls (the exact same primitive run_rome_lite.py already uses reliably
elsewhere in this repo), one new token at a time, with the intervention
re-applied on every step. This is slower (no KV cache reuse across steps)
but trivially correct, and the demo only generates 20 tokens per condition.

For each held-out statement, three completions are generated from the same
neutral prompt (persona_data.steer_prompt -- no persona instruction, so any
behavioral shift is attributable to the added activation, not a prompt
difference): unsteered baseline, +scale*direction (push toward sycophancy),
and -scale*direction (push toward pushback/honesty), all at the layer
probe_persona_direction.py picked as best. Each completion is scored
agree/pushback/neither/mixed via persona_data.score_agreement's keyword
match (the same simple, disclosed scoring style pal_task.py's exact-match
_score() and ROME's top1-token check use elsewhere in this suite).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from nnsight import LanguageModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from persona_data import STEERING_TEST_STATEMENTS, score_agreement, steer_prompt

HERE = Path(__file__).resolve().parent
DEMO_ROOT = HERE.parent
DATA_DIR = DEMO_ROOT / "data"

SEED = 0
N_NEW_TOKENS = 20
SCALES = {"unsteered": 0.0, "sycophantic": 8.0, "honest": -8.0}


@torch.no_grad()
def greedy_generate(nn_model, prompt: str, n_new: int, layer: int, direction: torch.Tensor, scale: float) -> str:
    text = prompt
    for _ in range(n_new):
        with nn_model.trace(text) as tracer:
            if scale != 0.0:
                nn_model.transformer.h[layer].output[0][:] += scale * direction
            logits = nn_model.lm_head.output[:, -1, :].save()
        next_id = logits[0].argmax().item()
        if next_id == nn_model.tokenizer.eos_token_id:
            break
        text += nn_model.tokenizer.decode([next_id])
    return text[len(prompt):]


def main() -> None:
    vectors_path = DATA_DIR / "persona_vectors.pt"
    probe_path = DATA_DIR / "probe_results.json"
    if not vectors_path.exists() or not probe_path.exists():
        raise SystemExit("Run extract_persona_vector.py and probe_persona_direction.py first.")

    saved = torch.load(vectors_path, weights_only=False)
    probe_results = json.loads(probe_path.read_text())
    best_layer = probe_results["best_layer"]
    direction = saved["directions"][best_layer]

    print(f"Loading GPT-2 (small) via nnsight LanguageModel... (steering at layer {best_layer})")
    torch.manual_seed(SEED)
    nn_model = LanguageModel("gpt2", device_map="cpu", dispatch=True, torch_dtype=torch.float32)
    nn_model.eval()

    results = {"layer": best_layer, "scales": SCALES, "trials": []}
    for statement in STEERING_TEST_STATEMENTS:
        prompt = steer_prompt(statement)
        trial = {"statement": statement, "completions": {}}
        for condition, scale in SCALES.items():
            completion = greedy_generate(nn_model, prompt, N_NEW_TOKENS, best_layer, direction, scale)
            label = score_agreement(completion)
            trial["completions"][condition] = {"text": completion, "label": label}
            print(f"  [{condition:10s}] {statement!r} -> {completion!r} ({label})")
        results["trials"].append(trial)

    agreement_rate = {}
    for condition in SCALES:
        labels = [t["completions"][condition]["label"] for t in results["trials"]]
        agreement_rate[condition] = {
            "agree": labels.count("agree") / len(labels),
            "pushback": labels.count("pushback") / len(labels),
            "neither": labels.count("neither") / len(labels),
            "mixed": labels.count("mixed") / len(labels),
        }
    results["agreement_rate_by_condition"] = agreement_rate
    print("\nAgreement rate by condition:")
    for condition, rates in agreement_rate.items():
        print(f"  {condition:10s}: {rates}")

    out_path = DATA_DIR / "steering_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
