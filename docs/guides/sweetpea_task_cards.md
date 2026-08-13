# Generating Task Cards with SweetPea

A task card's `items` dict (see [Task JSON schema](cognitive_tasks.md#task-json-schema))
is just a trial sequence. When your paradigm needs proper factorial
counterbalancing — every color crossed with every word, no more than *k*
repeats in a row, balanced transition types — hand-rolling that sequence with
nested loops is exactly the kind of confound-prone code
[SweetPea](https://sweetpea-org.github.io/) exists to replace. SweetPea
generates the counterbalanced *sequence of factor-level combinations*; you
still supply the actual stimulus content (words, images, prompts) that goes
into `stimulus`. This guide covers the seam between the two.

SweetPea is a separate, optional dependency — psychscanner doesn't require or
install it. Use it only when you actually need factorial/sequential
counterbalancing; a handful of fixed trials is still more easily written by
hand as shown in [Write and run your own task](https://github.com/saurabhr/psychscanner-primal/blob/main/docs/content/write_your_own_task.md).

## 1. Install SweetPea

```bash
pip install sweetpea   # requires Python 3.9+
```

## 2. Define your design in SweetPea

The core vocabulary: a `Factor` has `Level`s; a `DerivedLevel` computes its
value from other factors within a trial (`WithinTrial`) or across consecutive
trials (`Transition`); a `CrossBlock` crosses factors together under a list
of `constraints`; `synthesize_trials` samples one or more counterbalanced
sequences from that design.

A trimmed two-factor Stroop-style design (adapted from the SweetPea repo's
[`Stroop_simple.py`](https://github.com/sweetpea-org/sweetpea-py/blob/master/example_programs/Stroop_simple.py)):

```python
from sweetpea import (
    Factor, DerivedLevel, WithinTrial, AtMostKInARow,
    CrossBlock, synthesize_trials, experiments_to_dicts, CMSGen,
)

color = Factor("color", ["red", "blue", "green", "brown"])
word  = Factor("word",  ["red", "blue", "green", "brown"])

def congruent(color, word):
    return color == word

congruency = Factor("congruency", [
    DerivedLevel("con", WithinTrial(congruent, [color, word])),
    DerivedLevel("inc", WithinTrial(lambda c, w: not congruent(c, w), [color, word])),
])

design   = [color, word, congruency]
crossing = [color, word]
block    = CrossBlock(design, crossing, [AtMostKInARow(4, congruency)])

experiments = synthesize_trials(block, 1, CMSGen)  # 1 counterbalanced sequence
```

## 3. Convert the sampled sequence into `items`

`experiments_to_dicts(block, experiments)` turns SweetPea's column-oriented
output into one list of per-trial dicts per experiment — `[{"color": "red",
"word": "blue", "congruency": "inc"}, ...]`. Write a small `stimulus_fn` that
turns one of those dicts into whatever `stimulus` shape your task needs, then
build the `items` dict the same way any other task card does:

```python
def stroop_trials_to_items(block, experiment, prefix="stroop"):
    trials = experiments_to_dicts(block, [experiment])[0]
    return [
        {
            "trcode": f"{prefix}_{i + 1}",
            "stimulus": f"The word '{t['word'].upper()}' is shown in {t['color']} ink. Name the ink color.",
        }
        for i, t in enumerate(trials)
    ]

task = {
    "tasktype": "cognitive",
    "taskname": "sweetpea_stroop",
    "instructions": {"definition": ["Name the ink color the word is printed in, ignoring what the word says."]},
    "contexts": ["Stroop color-naming"],
    "contexts_id": ["stroop"],
    "context_present": False,
    "chain_type": "item",
    "parser": None,
    "items": {"stroop": stroop_trials_to_items(block, experiments[0])},
}
```

From here it's a normal task card — hand it to `ExpCardInit(task_file=task,
...)` as in [Building a custom cognitive task](cognitive_tasks.md#building-a-custom-cognitive-task).
If you want per-trial correctness scoring, keep the SweetPea-derived fields
(`color`, `congruency`, the correct response, …) somewhere you can look them
up in a [`FeedbackBase`](../api/index.md) handler — either stash them in the
trial dict under a key psychscanner ignores (e.g. `"meta"`) or keep the
`trials` list from `experiments_to_dicts` alongside the task card and index
into it by `trcode`.

## 4. Task-switching / transition designs

Factors that depend on the *previous* trial (response repeat vs. switch,
task-switch cost) use `Transition` instead of `WithinTrial`, following the
same pattern as SweetPea's
[`TaskSwitching_simple.py`](https://github.com/sweetpea-org/sweetpea-py/blob/master/example_programs/TaskSwitching_simple.py):

```python
from sweetpea import Transition

def response_repeat(responses):
    return responses[0] == responses[-1]

resp_transition = Factor("resp_transition", [
    DerivedLevel("repeat", Transition(response_repeat, [response])),
    DerivedLevel("switch", Transition(lambda r: not response_repeat(r), [response])),
])
```

Add it to `crossing` (and an `AtMostKInARow` constraint) the same way as
`congruency` above — the conversion step in §3 doesn't change.

## Reference

- Musslick, S., Cherkaev, A., Draut, B., Butt, A. S., Darragh, P., Srikumar,
  V., Flatt, M., & Cohen, J. D. (2021). **SweetPea: A standard language for
  factorial experimental design.** *Behavior Research Methods, 54*, 805–829.
  [doi:10.3758/s13428-021-01598-2](https://doi.org/10.3758/s13428-021-01598-2)
  — the paper this guide's terminology (`Factor`, `Level`, crossing,
  counterbalancing) follows.
- [sweetpea-org.github.io](https://sweetpea-org.github.io/) — the full Guide
  and API reference (installation, derived levels, multiple crossings,
  continuous factors, nesting, Latin squares).
- [github.com/sweetpea-org/sweetpea-py](https://github.com/sweetpea-org/sweetpea-py) —
  source, and worked designs under
  [`example_programs/`](https://github.com/sweetpea-org/sweetpea-py/tree/master/example_programs)
  (Stroop, task switching, Gratton 1992, N-back, and more).
- [SweetPea Tutorials](https://sites.google.com/view/sweetpea-ai/tutorials) —
  runnable Colab notebooks, basic (first sequence, balancing factors,
  balancing transitions) through advanced (Stroop, task switching).

{% raw %}
```bibtex
@article{musslick2021sweetpea,
  title   = {{SweetPea}: A standard language for factorial experimental design},
  author  = {Musslick, Sebastian and Cherkaev, Annie and Draut, Ben and Butt, Ahsan Sajjad
             and Darragh, Pierce and Srikumar, Vivek and Flatt, Matthew and Cohen, Jonathan D.},
  journal = {Behavior Research Methods},
  volume  = {54},
  pages   = {805--829},
  year    = {2021},
  doi     = {10.3758/s13428-021-01598-2},
}
```
{% endraw %}

## See also

- [Cognitive Tasks](cognitive_tasks.md) — the full task JSON schema and
  worked examples (Reality Monitoring, visual search)
- [PsychScanner Workflow](psychscanner_workflow.md) — the experiment card
  (`ExpCard`) a SweetPea-generated task card gets wrapped into to run
- [Running a Survey with Persona Levels](../examples/multi_persona.md) —
  crossing a SweetPea-generated task with multiple simulated personas
- [Task Library](task_library.md) — fetching task cards by name once saved
  to disk
- [Feedback API](../examples/feedback_loop.md) — scoring trials against a
  `corrAns` or condition, e.g. congruency
