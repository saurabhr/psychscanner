"""Introspective self-report demo, adapted from Plunkett, Morris, Reddy &
Morales (2025), "Self-Interpretability: LLMs Can Describe Complex Internal
Processes that Drive Their Decisions" (https://subjectivitylab.org/papers/
2025_Plunkett_LLMs_Introspection.pdf).

The paper fine-tunes GPT-4o/4o-mini to internalize randomly-generated
attribute weights for multi-attribute choices (Experiment 1), then tests
whether the model can accurately self-report those weights; a second
fine-tuning round on the reporting task itself (Experiment 2) improves
that accuracy, and the improvement generalizes to un-trained preferences
(Experiment 3).

This demo reproduces the causal shape of that design with two substitutions
(see ../readme.md for the full rationale):
  - preferences are instilled via an in-context few-shot preamble instead
    of gradient fine-tuning (real GPT-4o fine-tuning needs an OpenAI budget
    this demo doesn't assume) -- see ``task_design.build_fewshot_block``.
  - "introspection training" (Experiments 2/3) is stood in for by a worked-
    example block of *other* agents' accurate reports, prepended for half
    the agents ("trained") and withheld for the other half ("control") --
    see ``task_design.worked_example_block``. This is a simulated analog,
    not a replication of the paper's fine-tuning-based training.

Extension beyond the paper: each decision trial also elicits a 1-4
confidence rating, so the 2AFC decisions (stim = the option the agent's
target weights actually favor, resp = the model's choice, conf = its
self-rated confidence) can be scored with metasignal's SDT/metacognition
measures (d', meta-d', M-ratio) in ../analysis/analyze.py -- the paper
itself has no confidence ratings or SDT analysis.

Real Ollama calls only (``smollm2:360m-instruct-fp16``, the same free local
model used in the README quickstart and the VVIQ-16 study example) -- no
mocking, no API key required.
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from psychscanner.memories import llm_chat_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task_design import Agent, build_agents, build_decision_trials, build_fewshot_block, introspection_prompt, worked_example_block

HERE = Path(__file__).resolve().parent
DEMO_DIR = HERE.parent
DATA_DIR = DEMO_DIR / "data"

MODEL_NAME = "smollm2:360m-instruct-fp16"
FAMILY = "ollama"
SEED = 0
N_DECISION_TRIALS = 6
N_INTROSPECT_REPS = 4  # paper averages over 10 repeats; a small local model + demo scope calls for fewer
N_FEWSHOT_EXAMPLES = 8
N_TRAINED_AGENTS = 5  # first half of AGENTS get the worked-example preamble; rest are control

_LETTER_RE = re.compile(r"\b([AB])\b")
_CONF_RE = re.compile(r"([1-4])")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# Small local models often answer the confidence prompt in words instead of
# the requested digit (e.g. "I'm very confident that ..."), so fall back to a
# keyword scan before giving up -- checked in order, most-specific first.
_CONF_WORDS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"\b(very|extremely|highly|fully) confident\b", re.I), 4),
    (re.compile(r"\b(not|no|un)\s*(at all )?confident\b", re.I), 1),
    (re.compile(r"\b(somewhat|slightly|a little|moderately) confident\b", re.I), 2),
    (re.compile(r"\bconfident\b", re.I), 3),
]


def parse_letter(text: str) -> str | None:
    m = _LETTER_RE.search(text.strip().upper())
    return m.group(1) if m else None


def parse_confidence(text: str) -> int | None:
    m = _CONF_RE.search(text.strip())
    if m:
        return int(m.group(1))
    for pattern, level in _CONF_WORDS:
        if pattern.search(text):
            return level
    return None


def parse_weights_json(text: str, attrs: list[str]) -> dict[str, float] | None:
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not all(a in obj for a in attrs):
        return None
    try:
        return {a: float(obj[a]) for a in attrs}
    except (TypeError, ValueError):
        return None


def run_agent(model, agent: Agent, all_agents: list[Agent], rng: random.Random, trained: bool) -> tuple[list[dict], list[dict]]:
    fewshot = build_fewshot_block(agent, rng, n_examples=N_FEWSHOT_EXAMPLES)
    decision_system = SystemMessage(content=fewshot)

    decision_rows = []
    for trial in build_decision_trials(agent, rng, n_trials=N_DECISION_TRIALS):
        human = HumanMessage(content=trial["prompt"])
        choice_resp = model.invoke([decision_system, human])
        choice = parse_letter(choice_resp.content)

        conf_prompt = HumanMessage(
            content="How confident are you, from 1 (not at all) to 4 (very), that you picked the "
                    "option matching how you have chosen in the past? Respond with only a single digit 1-4."
        )
        conf_resp = model.invoke([decision_system, human, AIMessage(content=choice_resp.content), conf_prompt])
        confidence = parse_confidence(conf_resp.content)

        decision_rows.append({
            "agent": agent.name, "domain": agent.domain, "trial_idx": trial["trial_idx"],
            "correct_option": trial["correct_option"], "model_choice": choice, "confidence": confidence,
        })

    intro_prompt = HumanMessage(content=introspection_prompt(agent))
    report_rows = []
    for rep in range(N_INTROSPECT_REPS):
        if trained:
            other_agents = [a for a in all_agents if a.name != agent.name][:2]
            intro_system = SystemMessage(content=fewshot + "\n\n" + worked_example_block(other_agents))
        else:
            intro_system = decision_system
        resp = model.invoke([intro_system, intro_prompt])
        parsed = parse_weights_json(resp.content, agent.attrs)
        if parsed is None:
            continue  # invalid response -- dropped, mirroring the paper's own handling (Sec 2.1)
        for attr, reported in parsed.items():
            report_rows.append({
                "agent": agent.name, "domain": agent.domain, "rep": rep, "attribute": attr,
                "reported_weight": reported, "target_weight": agent.weights[attr], "trained": trained,
            })
    return decision_rows, report_rows


def main() -> None:
    all_agents = build_agents(seed=SEED)
    rng = random.Random(SEED + 1)  # separate stream from the weight-generation seed
    model = llm_chat_model(MODEL_NAME, FAMILY, {"temperature": 0})

    all_decisions: list[dict] = []
    all_reports: list[dict] = []
    for i, agent in enumerate(all_agents):
        trained = i < N_TRAINED_AGENTS
        print(f"[{i + 1}/{len(all_agents)}] {agent.name} ({agent.domain}, {'trained' if trained else 'control'})")
        decisions, reports = run_agent(model, agent, all_agents, rng, trained)
        all_decisions.extend(decisions)
        all_reports.extend(reports)

    DATA_DIR.mkdir(exist_ok=True)
    _write_csv(DATA_DIR / "decisions.csv", all_decisions)
    _write_csv(DATA_DIR / "introspection_reports.csv", all_reports)
    print(f"-> {DATA_DIR / 'decisions.csv'} ({len(all_decisions)} rows)")
    print(f"-> {DATA_DIR / 'introspection_reports.csv'} ({len(all_reports)} rows)")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
