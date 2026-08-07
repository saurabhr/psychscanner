"""Run the genuine paired-associate-learning (PAL) task — examples/tasks/pal50.json,
tasktype "episodic" — through all 5 psychscanner features this demo folder exists
to show, on real model calls:

    1. Single-Turn     -- memory="SingleTurn"                     (no carry-over at all)
    2. Trial-Chain     -- memory="Convo", chain_type="trial"       (per-pair thread)
    3. Episodic-Chain  -- memory="Convo", chain_type="task"        (whole-session thread)
    4. Correct/Incorrect Feedback -- episodic-chain + CorrectIncorrectFeedback
    5. Reward Feedback            -- episodic-chain + RewardFeedback

This is a paired-associate memory task (NOT Reality Monitoring — no rm_*.json
file is used here). See ``pal_task.py`` for how the 5-condition task shapes are
built from pal50.json and for the two FeedbackBase handlers.

Run with real Groq + OpenRouter API keys (`GROQ_API_KEY`, `OPENROUTER_API_KEY`
in `.env` at repo root):

    source .venv311/bin/activate   # this is the env with langchain-groq/-openai installed
    python demonstrations/02_association_memory/simulation/run_simulation.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from psychscanner import ExpCard, ExpCardInit, ScannerModel, to_csv  # noqa: E402
from pal_task import (  # noqa: E402
    PAIRS_PER_LEVEL,
    CorrectIncorrectFeedback,
    RewardFeedback,
    build_subset,
)

DEMO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = DEMO_DIR / "data"
RUN_DIR = DATA_DIR / "_runs"  # raw .psyscan session files

# Two providers already configured in .env — a Groq model (structured-output
# capable, unlike llama-3.1-8b-instant which 400s on response_format=json_schema)
# and an OpenRouter model, for a cross-model comparison point.
MODELS = [
    ("openai/gpt-oss-120b", "groq"),
    ("meta-llama/llama-3.1-8b-instruct", "openrouter"),
]

CONDITIONS = ["single_turn", "trial_chain", "episodic_chain", "feedback_ci", "feedback_reward"]


def build_card(condition: str, model: str, family: str) -> ExpCardInit:
    card = ExpCardInit()
    card.model, card.family = model, family
    card.parameters = {"temperature": 0}
    card.parser = "0"  # per-trial JSON "parser" field routes: None -> unstructured (study), "PairedAssociateRecall" -> structured (test)
    card.cogtype, card.nsim = "no", 1
    card.proj_dir, card.projectname = RUN_DIR, f"pal_{condition}"
    card.tunnel_status = "0"

    if condition == "single_turn":
        card.task_file = build_subset(PAIRS_PER_LEVEL, share_trcode=False)
        card.memory, card.chain_type = "SingleTurn", "item"
    elif condition == "trial_chain":
        card.task_file = build_subset(PAIRS_PER_LEVEL, share_trcode=True)
        card.memory, card.chain_type = "Convo", "trial"
    elif condition == "episodic_chain":
        card.task_file = build_subset(PAIRS_PER_LEVEL, share_trcode=False)
        card.memory, card.chain_type = "Convo", "task"
    elif condition == "feedback_ci":
        card.task_file = build_subset(PAIRS_PER_LEVEL, share_trcode=False)
        card.memory, card.chain_type = "Convo", "task"
        card.feedback, card.feedback_fn = True, CorrectIncorrectFeedback
    elif condition == "feedback_reward":
        card.task_file = build_subset(PAIRS_PER_LEVEL, share_trcode=False)
        card.memory, card.chain_type = "Convo", "task"
        card.feedback, card.feedback_fn = True, RewardFeedback
    else:
        raise ValueError(f"unknown condition: {condition}")
    return card


def run_one(condition: str, model: str, family: str) -> Path:
    card = build_card(condition, model, family)
    scanner = ScannerModel(expcard=ExpCard(card))
    scanner.run()
    model_tag = model.replace("/", "-")
    out_csv = DATA_DIR / f"{condition}__{family}-{model_tag}.csv"
    to_csv(scanner, path=out_csv)
    return out_csv


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for model, family in MODELS:
        for condition in CONDITIONS:
            print(f"\n==== {condition} :: {family}/{model} ====", flush=True)
            t1 = time.time()
            out = run_one(condition, model, family)
            print(f"saved {out}  ({time.time()-t1:.0f}s)", flush=True)
    print(f"\nTotal runtime: {time.time()-t0:.0f}s")
