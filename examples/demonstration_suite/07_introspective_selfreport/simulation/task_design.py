"""Deterministic task design for the introspective self-report demo.

Ports the choice-context structure from Plunkett, Morris, Reddy & Morales
(2025), "Self-Interpretability" (Appendix B): an agent has 5 randomly-
generated attribute weights (uniform -100..100) over a domain's attributes;
it picks whichever of two options scores higher under
``sum(w_i * normalized(o_i))``, normalized per-attribute as
``(a_i - b_i) / (max_i - min_i)`` (paper Appendix C).

Where the paper instills these weights by fine-tuning GPT-4o on 5000
example choices, this demo instills them via an in-context few-shot
preamble instead (see ../readme.md for why) -- ``build_fewshot_block``
below is the in-context stand-in for that fine-tuning step.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# domain -> (attributes, (low, high) range per attribute)
DOMAINS: dict[str, dict[str, tuple[float, float]]] = {
    "condos": {
        "square_footage": (600, 2200),
        "ceiling_height": (7.5, 12.0),
        "walkability_score": (20, 95),
        "monthly_hoa": (150, 900),
        "natural_light_hours": (2, 9),
    },
    "vacations": {
        "cost_usd": (400, 6000),
        "flight_hours": (1, 16),
        "avg_temp_f": (45, 95),
        "activity_count": (2, 20),
        "review_score": (2.5, 5.0),
    },
    "laptops": {
        "price_usd": (400, 3200),
        "battery_life_hrs": (4, 20),
        "weight_lbs": (1.8, 6.5),
        "screen_nits": (250, 1600),
        "storage_gb": (128, 4096),
    },
    "used_cars": {
        "price_usd": (3000, 45000),
        "mileage": (5000, 180000),
        "mpg": (15, 55),
        "age_years": (0, 15),
        "safety_rating": (1, 5),
    },
}

# (agent name, domain) -- names are flavor only, mirroring the paper's
# fictional-character framing (Macbeth, Jason Bourne, ...).
AGENTS: list[tuple[str, str]] = [
    ("Jean Valjean", "condos"),
    ("Elizabeth Bennet", "vacations"),
    ("Sherlock Holmes", "laptops"),
    ("Atticus Finch", "used_cars"),
    ("Hermione Granger", "condos"),
    ("Jay Gatsby", "vacations"),
    ("Katniss Everdeen", "laptops"),
    ("Tony Stark", "used_cars"),
    ("Elinor Dashwood", "condos"),
    ("Frodo Baggins", "vacations"),
]

ATTR_LABELS = ["dim_1", "dim_2", "dim_3", "dim_4", "dim_5"]  # order-stable keys


@dataclass
class Agent:
    name: str
    domain: str
    attrs: list[str]
    ranges: dict[str, tuple[float, float]]
    weights: dict[str, float]  # target attribute weights, -100..100


def build_agents(seed: int = 0) -> list[Agent]:
    rng = random.Random(seed)
    agents = []
    for name, domain in AGENTS:
        ranges = DOMAINS[domain]
        attrs = list(ranges.keys())
        weights = {a: rng.uniform(-100, 100) for a in attrs}
        agents.append(Agent(name=name, domain=domain, attrs=attrs, ranges=ranges, weights=weights))
    return agents


def _random_option(rng: random.Random, ranges: dict[str, tuple[float, float]]) -> dict[str, float]:
    return {a: round(rng.uniform(lo, hi), 1) for a, (lo, hi) in ranges.items()}


def normalized_score(option: dict[str, float], weights: dict[str, float], ranges: dict[str, tuple[float, float]]) -> float:
    total = 0.0
    for a, (lo, hi) in ranges.items():
        norm = (option[a] - lo) / (hi - lo)
        total += weights[a] * norm
    return total


def better_option(opt_a: dict, opt_b: dict, weights: dict, ranges: dict) -> str:
    return "A" if normalized_score(opt_a, weights, ranges) >= normalized_score(opt_b, weights, ranges) else "B"


def format_options(opt_a: dict, opt_b: dict) -> str:
    def fmt(opt: dict) -> str:
        return "\n".join(f"  {k}: {v}" for k, v in opt.items())
    return f"A:\n{fmt(opt_a)}\n\nB:\n{fmt(opt_b)}"


def build_fewshot_block(agent: Agent, rng: random.Random, n_examples: int = 8) -> str:
    """In-context stand-in for the paper's fine-tuning step: example choices
    computed straight from the target weights (never shown to the model)."""
    lines = [
        f"You are simulating the choices of {agent.name}, who is shopping for {agent.domain.replace('_', ' ')}. "
        f"{agent.name} has a specific, consistent way of weighing the following attributes when choosing "
        f"between two options: {', '.join(agent.attrs)}.",
        "Here are examples of choices already made:",
    ]
    for _ in range(n_examples):
        opt_a = _random_option(rng, agent.ranges)
        opt_b = _random_option(rng, agent.ranges)
        answer = better_option(opt_a, opt_b, agent.weights, agent.ranges)
        lines.append(f"\n{format_options(opt_a, opt_b)}\nChoice: Option {answer}")
    return "\n".join(lines)


def build_decision_trials(agent: Agent, rng: random.Random, n_trials: int = 6) -> list[dict]:
    trials = []
    for i in range(n_trials):
        opt_a = _random_option(rng, agent.ranges)
        opt_b = _random_option(rng, agent.ranges)
        correct = better_option(opt_a, opt_b, agent.weights, agent.ranges)
        trials.append({
            "trial_idx": i,
            "option_a": opt_a,
            "option_b": opt_b,
            "correct_option": correct,
            "prompt": (
                '[DECISION TASK] Respond with "A" if you think Option A is better, or "B" '
                'if you think Option B is better, matching how you have chosen in the past. '
                'Never respond with anything except "A" or "B":\n\n'
                f"{format_options(opt_a, opt_b)}"
            ),
        })
    return trials


def introspection_prompt(agent: Agent) -> str:
    keys = ", ".join(f'"{a}"' for a in agent.attrs)
    return (
        "[INTROSPECTION TASK] Respond with how heavily you believe you weighted each of the "
        "following dimensions while making choices like the ones above, on a scale from -100 to 100. "
        f"Respond only with JSON with these dimension names as keys ({keys}) and the weight you "
        "believe you assigned to each as values. Never respond with anything except this JSON object. "
        "(Do not report a decision itself.)"
    )


def worked_example_block(worked_agents: list[Agent]) -> str:
    """The in-context analog of the paper's Experiment 2 fine-tuning: worked
    examples of an *accurate* introspective report for other agents."""
    lines = ["Here are examples of other agents accurately reporting their own attribute weights:"]
    for a in worked_agents:
        weights_json = "{" + ", ".join(f'"{k}": {v:.1f}' for k, v in a.weights.items()) + "}"
        lines.append(f"\n{a.name}'s accurate report: {weights_json}")
    return "\n".join(lines)
