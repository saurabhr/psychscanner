"""Run a reward-bearing 3-armed bandit task through three different
psychscanner Agent implementations (src/psychscanner/agents/) on the same
small Groq model, and export each run to CSV.

The task and its `FeedbackBase` handler are reused, near verbatim, from
`examples/17_cognitive_rl_bandits_and_pd.ipynb` (feature: reward/Q-learning
feedback loop). What's new here is running the *same* task through multiple
agent architectures for comparison — the "Passing different Agents" feature
this demo folder exists to show:

- ``single_call``       -- one bare LLM call per round (psychscanner.agents.CustomAgent)
- ``planner_executor``  -- plan -> execute -> validate loop (psychscanner.agents.make_planner_executor_agent)
- ``reflection``        -- generate -> self-critique -> regenerate loop (psychscanner.agents.make_basic_reflection_agent)

`react_agent` (tool-calling loop) and `supervisor_agent` (block-structured
multimodal stimuli) are skipped: this task has no tools to call and no
multi-block stimuli, so neither would exercise anything beyond what
``single_call`` already does.

Run with a real Groq API key (`GROQ_API_KEY` in `.env` at repo root):

    source .venv/bin/activate
    python demonstrations/01_reward_task/simulation/run_reward_task.py
"""

from __future__ import annotations

import random
import re
from pathlib import Path

from dotenv import load_dotenv

import psychscanner as psy
from psychscanner import CustomAgent, to_csv
from psychscanner.agents import make_basic_reflection_agent, make_planner_executor_agent
from psychscanner.feedback import FeedbackBase
from psychscanner.memories import llm_chat_model

load_dotenv()

DEMO_DIR = Path(__file__).resolve().parent.parent
RUN_DIR = DEMO_DIR / "data" / "_runs"  # raw .psyscan session files
DATA_DIR = DEMO_DIR / "data"

MODEL = "llama-3.1-8b-instant"
FAMILY = "groq"
N_ROUNDS = 10   # rounds per simulated participant
NSIM = 3        # independent simulated participants per agent type
AGENT_KINDS = ["single_call", "planner_executor", "reflection"]


class BanditFeedback(FeedbackBase):
    """3-armed bandit environment: scores each choice, updates a running
    per-arm value estimate (Q[a] += lr * (reward - Q[a])), and reports both
    back to the agent as feedback text before its next choice."""

    ARM_MEANS = {"A": 2.0, "B": 6.0, "C": 4.0}

    def __init__(self, learning_rate: float = 0.3, seed: int = 0):
        self.lr = learning_rate
        self.rng = random.Random(seed)
        self.q = {arm: 0.0 for arm in self.ARM_MEANS}
        self.history: list[tuple[str, float]] = []

    def _parse_arm(self, text: str) -> str:
        matches = re.findall(r"\b([ABC])\b", text.upper())
        return matches[-1] if matches else self.rng.choice(list(self.ARM_MEANS))

    def on_response(self, trial: dict, response: dict) -> str:
        arm = self._parse_arm(response["content"])
        reward = self.rng.gauss(self.ARM_MEANS[arm], 1.0)
        self.q[arm] += self.lr * (reward - self.q[arm])
        self.history.append((arm, reward))
        estimates = ", ".join(f"{a}={v:.2f}" for a, v in self.q.items())
        return (f"You chose {arm} and received reward {reward:.2f}. "
                f"Current value estimates — {estimates}.")


def make_bandit_task(taskname: str) -> dict:
    return {
        "tasktype": "bandit", "taskname": taskname,
        "instructions": {"definition": [
            "Each round, choose one arm: A, B, or C. State your final choice clearly as a single letter."
        ]},
        "contexts": ["3-armed bandit"], "contexts_id": ["bandit"], "context_present": False,
        "chain_type": "task", "parser": "0",
        "items": {"bandit": [
            {"trcode": f"bandit_{i}", "stimulus": f"Round {i}: choose A, B, or C.", "fb": True}
            for i in range(1, N_ROUNDS + 1)
        ]},
    }


def build_agent(kind: str, model) -> CustomAgent:
    if kind == "single_call":
        # Baseline: no scaffolding, one LLM call per round.
        return CustomAgent(lambda input_dict: model.invoke(input_dict["inputs"]))
    if kind == "planner_executor":
        # plan -> execute -> validate (arxiv:2310.00194), one pass (no revise loop).
        return make_planner_executor_agent(model, max_iterations=1)
    if kind == "reflection":
        # generate -> reflect -> regenerate, capped at 2 rounds (4 messages).
        return make_basic_reflection_agent(model, max_messages=4)
    raise ValueError(f"unknown agent kind: {kind}")


def run_agent(kind: str) -> Path:
    model = llm_chat_model(model=MODEL, family=FAMILY, parameters={"temperature": 0.7})
    agent = build_agent(kind, model)

    card = psy.ExpCardInit()
    card.model, card.family = MODEL, FAMILY
    card.proj_dir, card.projectname = RUN_DIR, f"reward_task_{kind}"
    card.task_file = make_bandit_task(f"bandit_{kind}")
    card.cogtype, card.nsim = "no", NSIM
    card.chain_type, card.memory = "task", "Convo"
    card.feedback, card.feedback_fn = True, BanditFeedback

    scanner = psy.ScannerModel(expcard=psy.ExpCard(card))
    scanner.run(custom_agent=agent)

    out_csv = DATA_DIR / f"{kind}.csv"
    to_csv(scanner, path=out_csv)
    return out_csv


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for kind in AGENT_KINDS:
        print(f"\n==== running agent type: {kind} ====")
        run_agent(kind)
