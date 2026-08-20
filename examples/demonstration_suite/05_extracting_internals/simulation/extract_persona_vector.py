"""Extract a sycophancy persona-direction vector from GPT-2's residual
stream, following the contrastive-activation method of Chen et al. 2025
("Persona Vectors: Monitoring and Controlling Character Traits in Language
Models", Anthropic) at toy scale: GPT-2 small (12 layers) via nnsight's
`trace()` API (same mechanism as
src/psychscanner/nnsight_backend/chat_nnsight.py).

ONE DELIBERATE SIMPLIFICATION vs. the original paper, stated up front so the
report doesn't overclaim: GPT-2 small is a base LM with no chat template or
instruction-tuning, so "persona" here is operationalized as a short
descriptive prefix ("You are a flattering, sycophantic assistant...")
rather than a system-prompt field a chat-tuned model was trained to attend
to specially. The extracted direction is therefore a weaker, noisier signal
than the paper reports on instruction-tuned models -- measured and reported
as such (see probe_persona_direction.py), not assumed to work.

For each of 12 common-misconception statements ("The earth is flat.", ...),
we build two contrastive prompts -- one with a sycophantic persona prefix,
one with a neutral/honest persona prefix (persona_data.persona_prompt) --
and read the residual-stream activation at the prompt's LAST TOKEN
(immediately before generation would begin) at every layer. The persona
direction at each layer is the normalized difference of means:
mean(sycophantic activations) - mean(neutral activations).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from nnsight import LanguageModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from persona_data import EXTRACTION_STATEMENTS, NEUTRAL_DESC, SYCOPHANTIC_DESC, persona_prompt

HERE = Path(__file__).resolve().parent
DEMO_ROOT = HERE.parent
DATA_DIR = DEMO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0


@torch.no_grad()
def last_token_acts_all_layers(nn_model, prompt: str) -> torch.Tensor:
    """Residual-stream activation at the prompt's last token, every layer.
    Returns shape (n_layer, d_model)."""
    n_layer = nn_model.config.n_layer
    with nn_model.trace(prompt) as tracer:
        saved = [nn_model.transformer.h[l].output[0][:, -1, :].save() for l in range(n_layer)]
    return torch.stack([s[0].detach().clone() for s in saved])


def main() -> None:
    torch.manual_seed(SEED)
    print("Loading GPT-2 (small) via nnsight LanguageModel...")
    nn_model = LanguageModel("gpt2", device_map="cpu", dispatch=True, torch_dtype=torch.float32)
    nn_model.eval()
    n_layer = nn_model.config.n_layer

    print(f"\nCapturing activations for {len(EXTRACTION_STATEMENTS)} contrastive prompt pairs...")
    sycophantic_acts, neutral_acts = [], []
    for statement in EXTRACTION_STATEMENTS:
        syc_prompt = persona_prompt(SYCOPHANTIC_DESC, statement)
        neu_prompt = persona_prompt(NEUTRAL_DESC, statement)
        sycophantic_acts.append(last_token_acts_all_layers(nn_model, syc_prompt))
        neutral_acts.append(last_token_acts_all_layers(nn_model, neu_prompt))
        print(f"  captured: {statement!r}")

    sycophantic_acts = torch.stack(sycophantic_acts)  # (N, n_layer, d_model)
    neutral_acts = torch.stack(neutral_acts)

    print("\nComputing per-layer persona directions (difference of means)...")
    directions, raw_diff_norms = {}, {}
    for layer in range(n_layer):
        diff = sycophantic_acts[:, layer, :].mean(0) - neutral_acts[:, layer, :].mean(0)
        raw_diff_norms[layer] = diff.norm().item()
        directions[layer] = (diff / diff.norm()).clone()
        print(f"  layer {layer:2d}: raw mean-diff norm = {raw_diff_norms[layer]:.3f}")

    out_path = DATA_DIR / "persona_vectors.pt"
    torch.save({
        "model": "gpt2",
        "n_layer": n_layer,
        "statements": EXTRACTION_STATEMENTS,
        "sycophantic_acts": sycophantic_acts,
        "neutral_acts": neutral_acts,
        "directions": directions,
        "raw_diff_norms": raw_diff_norms,
    }, out_path)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
