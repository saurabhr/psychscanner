"""Prospect Theory gambles, three agent architectures from `src/psychscanner/agents/`.

Compares behavioral choice patterns across:
  - map              -- psychscanner.agents.map_agent.make_map_agent, the
                         Modular Agentic Planner (Webb, Mondal & Momennejad
                         2025, Nature Communications) built for this project.
                         Stateless per trial (matches the paper's own design
                         -- no cross-trial memory module).
  - convo_memory     -- a bare CustomAgent(model.invoke) run under
                         memory="Convo", so each trial's prompt carries the
                         running dialogue of prior gambles and choices. This
                         is "an agent with conversation memory" as its own
                         condition, not the MAP baseline -- MAP is
                         deliberately stateless per the paper.
  - lats             -- psychscanner.agents.reflection_agents.make_lats_agent,
                         already implementing Monte Carlo tree search over
                         candidate answers (the pattern described in
                         LangChain's reflection-agents post). Stateless per
                         trial, like MAP -- the comparison isolates search
                         strategy from memory.

Task: examples/tasks/prospect_theory_demo.tcard.psyscan (20 trials, built by
generators/generate_prospect_theory.py per musslick/experiment_factor_discovery's
benchmark spec). Real Groq calls (llama-3.1-8b-instant) throughout, no mocking
-- this file is the real run; the mock-llm smoke test already happened in
generate_prospect_theory.py's own verification step before this was written.

N_TRIALS below trims the 20-trial card to a smaller slice, purely to bound
API cost/time for this demo (each MAP/LATS trial makes several LLM calls,
not one) -- the full 20-trial ground truth CSV is unaffected and available
for a larger run.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from groq import RateLimitError

import psychscanner as psy
from psychscanner import load_task_card, to_csv
from psychscanner.agents import CustomAgent, make_lats_agent, make_map_agent
from psychscanner.memories import llm_chat_model

HERE = Path(__file__).resolve().parent
DEMO_DIR = HERE.parent
DATA_DIR = DEMO_DIR / "data"
RUN_DIR = HERE / "_run"
TASK_PATH = DEMO_DIR.parents[1] / "tasks" / "prospect_theory_demo.tcard.psyscan"

MODEL_NAME = "llama-3.1-8b-instant"
FAMILY = "groq"
N_TRIALS = 10  # of 20 -- see module docstring


class RetryModel:
    """Retry-on-429 wrapper, same as 01_reward_task/simulation/run_reward_task.py."""

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


def make_trimmed_task(n_trials: int = N_TRIALS) -> dict:
    card = load_task_card(TASK_PATH)
    keys = list(card["items"])[:n_trials]
    card["items"] = {k: card["items"][k] for k in keys}
    return card


def run_agent(name: str, agent, memory: str) -> None:
    print(f"\n=== running agent: {name} ===")
    card = psy.ExpCardInit()
    card.proj_dir, card.projectname = RUN_DIR, f"prospect_{name}"
    card.task_file = make_trimmed_task()
    card.cogtype, card.nsim = "no", 1
    card.chain_type, card.memory = "item", memory
    # custom_agent bypasses AgentConfig's LangChain model, but set these so
    # the exported CSV's model/family columns are honest.
    card.model, card.family = MODEL_NAME, FAMILY

    scanner = psy.ScannerModel(expcard=psy.ExpCard(card))
    scanner.run(custom_agent=agent)

    out_path = DATA_DIR / f"prospect_{name}.csv"
    df = to_csv(scanner, path=out_path)
    print(f"-> {len(df)} rows -> {out_path}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_model = llm_chat_model(model=MODEL_NAME, family=FAMILY, parameters={"temperature": 0.7})
    model = RetryModel(raw_model)

    agents = {
        "map": (make_map_agent(model), "SingleTurn"),
        "convo_memory": (CustomAgent(lambda d: model.invoke(d["inputs"])), "Convo"),
        "lats": (make_lats_agent(model, max_iterations=1, branching=2), "SingleTurn"),
    }

    for name, (agent, memory) in agents.items():
        run_agent(name, agent, memory)

    print("\nAll agents done. CSVs in", DATA_DIR)


if __name__ == "__main__":
    main()
