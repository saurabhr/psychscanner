# Example Task Definitions

Sample task JSON files for PsychScanner. Fetch any of them by name with
`psychscanner.task_library("name")`.

## Files

- `example_survey.json` - Simple survey template
- `bfi44.json` - Big Five Inventory personality survey (44 items)
- `nback_demo.json` - N-back working-memory task (n=1,2,3 routines), default task card — run with `memory="Convo"` (`memory_k=5` for conversation memory, or add `summary_k=10` for summary memory). **Known issue**: uses a `"routines"`/`"routines_flow"` schema that no code in `src/psychscanner` actually reads — can't currently run via `ExpCardInit`/`ScannerModel`. Use `nback_sweetpea_n2`–`n5` below instead.
- `pal50.json` - Paired-associate learning task (50 word pairs, study + test phases)
- `vviq16.json` - Vividness of Visual Imagery Questionnaire (16 items, simplified wording). See also the bundled `src/psychscanner/datasets/prompts/defaults/vviq.json` (taskname `vviqen`, official VVIQ wording) — kept as a separate variant, not merged.
- `rm_singleturn_demo.json` - Relational-memory task, single-turn trial
- `rm_trialchain_demo.json` - Relational-memory task, chained trials
- `rm_dynamic_demo.json` - Relational-memory task, dynamic encoding/test trials
- `rm_episodic_demo.json` - Relational-memory task, episodic conversation (no feedback)
- `rm_episodic_fb_demo.json` - Relational-memory task, episodic conversation with feedback

### SweetPea-generated (2026-08-13)

Real SweetPea generation via the scripts in `generators/`, not hand-written
trials — each generator's docstring says exactly what SweetPea generated vs.
what was computed in plain Python (and why, for the two designs where a
SweetPea derived-factor mechanism turned out unreliable on inspection and was
replaced). All four verified via a real mock-llm `ScannerModel` run plus an
independent re-derivation of every `corrAns` straight from the saved file.

- `nback_sweetpea_n2.tcard.psyscan` / `n3` / `n4` / `n5` - N-back, `generators/generate_nback_sweetpea.py`
- `stroop_sweetpea_demo.tcard.psyscan` - Stroop color-naming, `generators/generate_stroop_sweetpea.py`
- `taskswitch_sweetpea_demo.tcard.psyscan` - cued letter/number task-switching, `generators/generate_taskswitch_sweetpea.py`
- `vlm_shapes_demo.tcard.psyscan` - real images (color+shape naming), `generators/generate_vlm_shapes_demo.py`, extracted from `demonstration_suite/04_vlm_task`'s inline task

Not yet done: LLM-text vs. VLM-image *variants of the same three SweetPea
designs* (currently text-only) — `vlm_shapes_demo` above is a separate task,
not a visual version of Stroop/Task-Switching/N-back. Tracked in the
project's `TODO.md`.

## Structure

### Survey Task
```json
{
    "tasktype": "survey",
    "taskname": "your_survey",
    "instructions": {...},
    "items": {...},
    "parser": "ParserName"
}
```

### Cognitive Task
```json
{
    "tasktype": "sc",
    "taskname": "your_task",
    "instructions": {...},
    "items": {...},
    "chain_type": "trial"
}
```

## Using These Templates

```python
from psychscanner import ExpCard, ScannerModel

exp_card = ExpCard(
    model="gpt-3.5-turbo",
    family="openai",
    task_file="examples/tasks/example_survey.json",
    projectname="my_study"
)

scanner = ScannerModel(exp_card)
results = scanner.run()
```

## Creating Your Own

1. Copy a template
2. Modify instructions and items
3. Choose appropriate parser
4. Test with mock model first

See [Task Guide](../../docs/guides/survey_tasks.md) for details.

## Sharing a task card

Drop `<your_task_name>.json` directly into this directory (see [Structure](#structure)
above for the JSON shape) — `task_library("your_task_name")` and
`list_task_library()` pick it up immediately, no registration step, for local
use in your own checkout.

To contribute a card back as an official, vetted one, submit it to
[`psyscan-library`](https://github.com/saurabhr/psyscan-library) instead of
opening a PR against this folder — the public, versioned index of task and
experiment cards for psychscanner. Every submission there is validated
structurally, checked for duplicates, and actually run end-to-end against the
mock LLM before being merged. See that repo's `CONTRIBUTING.md` for the steps.
