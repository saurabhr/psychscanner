# Psychscanner Features Demonstrated

- 1: Direct chat-model invocation for a custom (non-task-card) protocol
  - Task: multi-attribute 2AFC choice + introspective self-report, adapted from Plunkett, Morris,
    Reddy & Morales (2025), ["Self-Interpretability: LLMs Can Describe Complex Internal Processes
    that Drive Their Decisions"](https://subjectivitylab.org/papers/2025_Plunkett_LLMs_Introspection.pdf).
    The paper fine-tunes GPT-4o/4o-mini to internalize random attribute weights, then tests whether
    the model can self-report those weights, and whether a second fine-tuning round on the
    self-reporting task itself improves accuracy and generalizes to un-trained preferences.
  - Two substitutions from the paper, both driven by this demo not assuming an OpenAI fine-tuning
    budget: preferences are instilled via an in-context few-shot preamble instead of gradient
    fine-tuning, and "introspection training" is stood in for by a worked-example block of *other*
    agents' accurate reports (prepended for half the agents, "trained"; withheld for the rest,
    "control") -- a simulated analog, not a replication, of the paper's Experiment 2/3 fine-tuning.
    See `reports/report.md` for the full comparison against the paper's own numbers.
  - Because the protocol (few-shot preamble -> decision trials -> introspection trials) doesn't fit
    the single-template task-card grammar, trials are driven directly via
    `psychscanner.memories.llm_chat_model` + `langchain_core.messages` (the same primitive
    `01_reward_task`'s baseline agent uses under the hood), not `ExpCard`/`ScannerModel`. Real
    Ollama calls only (`smollm2:360m-instruct-fp16`), no mocking. See `simulation/`.
  - Extension beyond the paper: each decision trial also elicits a 1-4 confidence rating, so the
    2AFC choices can be scored as a signal-detection problem via `metasignal`
    (stim = the option the agent's target weights actually favor, resp = the model's choice,
    conf = its self-rated confidence) -- d'/meta-d'/M-ratio measure how well the model's choices
    and confidence track its own instilled preferences. The paper has no confidence ratings or SDT
    analysis of its own; this is grounded in its decision-trial structure but goes beyond it. See
    `analysis/analyze.py`.
  - `psychscanner-primal` integration: the decision trials (the part with a real per-trial
    correct/incorrect signal -- primal's own inclusion bar) are also shipped as a standalone task
    card, `psychscanner-primal/examples/tasks/introspection_weights_demo.json`, runnable through
    primal's own `ExpCard`/`ScannerModel`/`task_library` quickstart pattern -- see
    `simulation/build_primal_task_card.py` and the note in that repo's
    `examples/tasks/README.md`. No new environment/`verifiers` package was added; the introspection
    (self-report) trials have no scalar ground truth to score, so they're intentionally left out of
    primal, consistent with its own "Excluded" list.

See `simulation/`, `data/`, `analysis/`, `reports/report.md`.
