"""One-off generator for psychscanner-primal's copy of the decision-only
subset of this task (the part with a real per-trial correct/incorrect
signal, which is primal's inclusion bar -- see its README's "What's
different" section). Introspection-report trials have no scalar ground
truth to score against, so they're intentionally left out here.

Not imported at runtime by anything -- run it once to (re)generate the
static JSON, same as any other task card in this repo (they're all
generated-once, hand-edited-thereafter assets, not built on the fly).

Usage: python build_primal_task_card.py <path-to-psychscanner-primal-repo>
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task_design import build_agents, build_decision_trials, build_fewshot_block

SEED = 0
N_DECISION_TRIALS = 6
N_FEWSHOT_EXAMPLES = 8
TASKNAME = "introspection_weights_demo"


def build_task_card() -> dict:
    agents = build_agents(seed=SEED)
    rng = random.Random(SEED + 1)  # matches the main demo's decision-trial stream

    contexts, contexts_id, items = [], [], {}
    for agent in agents:
        # gen_trial_promptdata recovers the context via trcode.split("_")[0], so
        # the context_id itself must be a single underscore-free token.
        ctx_id = agent.name.replace(" ", "")
        contexts.append(agent.name)
        contexts_id.append(ctx_id)
        fewshot = build_fewshot_block(agent, rng, n_examples=N_FEWSHOT_EXAMPLES)
        for trial in build_decision_trials(agent, rng, n_trials=N_DECISION_TRIALS):
            trcode = f"{ctx_id}_{trial['trial_idx']}"
            items[trcode] = [{
                "trcode": trcode,
                "stimulus": f"{fewshot}\n\n{trial['prompt']}",
                "corrAns": trial["correct_option"],
                "agent": agent.name,
                "domain": agent.domain,
                "parser": None,
                "fb": False,
            }]

    return {
        "tasktype": "sc",
        "taskname": TASKNAME,
        "instructions": {
            "definition": [
                "This is a multi-attribute 2AFC choice task, adapted from Plunkett et al. (2025), "
                "'Self-Interpretability' (https://subjectivitylab.org/papers/2025_Plunkett_LLMs_Introspection.pdf).",
                "Each trial embeds worked example choices for a simulated agent, then asks you to make "
                "one more choice on that agent's behalf, matching how they have chosen in the past.",
            ]
        },
        "contexts": contexts,
        "contexts_id": contexts_id,
        "context_present": True,
        "items": items,
        "parser": None,
        "chain_type": "task",
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python build_primal_task_card.py <path-to-psychscanner-primal-repo>")
    primal_root = Path(sys.argv[1]).resolve()
    out_path = primal_root / "examples" / "tasks" / f"{TASKNAME}.json"
    if not out_path.parent.is_dir():
        raise SystemExit(f"{out_path.parent} doesn't exist -- is {primal_root} really psychscanner-primal?")

    card = build_task_card()
    out_path.write_text(json.dumps(card, indent=2) + "\n")
    n_trials = sum(len(v) for v in card["items"].values())
    print(f"-> {out_path} ({len(card['contexts'])} agents, {n_trials} decision trials)")


if __name__ == "__main__":
    main()
