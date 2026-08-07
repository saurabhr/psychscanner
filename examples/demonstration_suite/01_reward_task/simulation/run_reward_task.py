"""Same reward task, three different Agent implementations from
``src/psychscanner/agents/`` — the "1: Passing different Agents" feature.

Task: the 3-armed bandit from ``examples/17_cognitive_rl_bandits_and_pd.ipynb``
(``BanditFeedback`` reused unchanged). Real API calls only, via Groq
(``llama-3.1-8b-instant`` — cheap/fast, no mocking).

Agents compared (one factory each, straight from psychscanner.agents):
  - custom_direct     -- psychscanner.agents.custom_agent.CustomAgent, a bare
                          single model.invoke() per trial (the baseline).
  - basic_reflection  -- psychscanner.agents.reflection_agents.make_basic_reflection_agent
                          (generate -> reflect -> generate loop).
  - planner_executor  -- psychscanner.agents.planner_executor_agent.make_planner_executor_agent
                          (plan -> execute -> validate loop, arxiv:2310.00194).

Each agent plays N_ROUNDS rounds of the bandit, NSIM simulated participants
each, under feedback=True + memory="Convo" so the running value estimates
carry across rounds within a participant -- exactly the ScannerModel/ExpCard
pipeline pattern from tutorial 17, just run through three Agent types
instead of one.
"""

from __future__ import annotations

import random
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from groq import RateLimitError

import psychscanner as psy
from psychscanner import to_csv
from psychscanner.agents import CustomAgent, make_basic_reflection_agent, make_planner_executor_agent
from psychscanner.feedback import FeedbackBase
from psychscanner.memories import llm_chat_model

HERE = Path(__file__).resolve().parent
DEMO_DIR = HERE.parent
DATA_DIR = DEMO_DIR / "data"
RUN_DIR = HERE / "_run"  # session/tunnel scratch dir, not the CSV output

MODEL_NAME = "llama-3.1-8b-instant"
FAMILY = "groq"
# Groq's free tier caps this model at 6000 tokens/min; a multi-call-per-trial
# agent (reflection, planner/executor) blows through that within a handful of
# rounds once the Convo-memory transcript grows. Keep the demo small rather
# than fight the quota with a bigger plan.
N_ROUNDS = 6
NSIM = 1  # simulated participants per agent type


class RetryModel:
    """Retry-on-429 wrapper around a LangChain chat model's .invoke().

    Reflection/planner-executor agents close over the model object and call
    ``model.invoke(...)`` directly inside their own LangGraph nodes, so this
    wraps that single method rather than touching agent internals.
    ponytail: fixed 25s backoff (Groq's TPM window is 60s), swap for
    Retry-After-header parsing if a cheaper backoff is ever needed.
    """

    def __init__(self, model, max_retries: int = 5, backoff_s: float = 25.0):
        self._model = model
        self.max_retries = max_retries
        self.backoff_s = backoff_s

    def invoke(self, *args, **kwargs):
        for attempt in range(self.max_retries + 1):
            try:
                return self._model.invoke(*args, **kwargs)
            except RateLimitError:
                if attempt == self.max_retries:
                    raise
                print(f"  [rate limited, retry {attempt + 1}/{self.max_retries} in {self.backoff_s:.0f}s]")
                time.sleep(self.backoff_s)


class BanditFeedback(FeedbackBase):
    """3-armed bandit environment, reused as-is from tutorial 17.

    Arms A/B/C have hidden means 2/6/4 (+ Gaussian noise). Q[a] += lr *
    (reward - Q[a]) is the whole "RL library" -- one instance per simulated
    participant (see FeedbackBase docstring), so cross-round state (running
    value estimates) lives safely in self.
    """

    ARM_MEANS = {"A": 2.0, "B": 6.0, "C": 4.0}

    def __init__(self, learning_rate=0.3, seed=0):
        self.lr = learning_rate
        self.rng = random.Random(seed)
        self.q = {arm: 0.0 for arm in self.ARM_MEANS}
        self.history = []  # (arm, reward) per round

    def _parse_arm(self, text: str) -> str:
        matches = re.findall(r"\b([ABC])\b", text.upper())
        return matches[-1] if matches else self.rng.choice(list(self.ARM_MEANS))

    def on_response(self, trial, response):
        arm = self._parse_arm(response["content"])
        reward = self.rng.gauss(self.ARM_MEANS[arm], 1.0)
        self.q[arm] += self.lr * (reward - self.q[arm])
        self.history.append((arm, reward))
        estimates = ", ".join(f"{a}={v:.2f}" for a, v in self.q.items())
        return (f"You chose {arm} and received reward {reward:.2f}. "
                f"Current value estimates - {estimates}.")


def make_bandit_task() -> dict:
    return {
        "tasktype": "bandit", "taskname": "cognitive_rl_bandit",
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


def run_agent(name: str, agent, model) -> None:
    print(f"\n=== running agent: {name} ===")
    card = psy.ExpCardInit()
    card.proj_dir, card.projectname = RUN_DIR, f"bandit_{name}"
    card.task_file = make_bandit_task()
    card.cogtype, card.nsim = "no", NSIM
    card.chain_type, card.memory = "task", "Convo"
    card.feedback, card.feedback_fn = True, BanditFeedback
    # custom_agent bypasses AgentConfig's LangChain model entirely, but set
    # these anyway so the exported CSV's model/family columns are honest
    # (otherwise they'd default to "mock-llm").
    card.model, card.family = MODEL_NAME, FAMILY

    scanner = psy.ScannerModel(expcard=psy.ExpCard(card))
    scanner.run(custom_agent=agent)

    out_path = DATA_DIR / f"bandit_{name}.csv"
    df = to_csv(scanner, path=out_path)
    print(f"-> {len(df)} rows -> {out_path}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_model = llm_chat_model(model=MODEL_NAME, family=FAMILY, parameters={"temperature": 0.7})
    model = RetryModel(raw_model)

    agents = {
        "custom_direct": CustomAgent(lambda d: model.invoke(d["inputs"])),
        "basic_reflection": make_basic_reflection_agent(model, max_messages=4),
        "planner_executor": make_planner_executor_agent(model, max_iterations=1),
    }

    for name, agent in agents.items():
        run_agent(name, agent, model)

    print("\nAll agents done. CSVs in", DATA_DIR)


if __name__ == "__main__":
    main()
