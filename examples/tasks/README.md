# Example Task Definitions

Sample task JSON files for PsychScanner. Fetch any of them by name with
`psychscanner.task_library("name")`.

## Files

- `example_survey.json` - Simple survey template
- `bfi44.json` - Big Five Inventory personality survey (44 items)
- `nback_demo.json` - N-back working-memory task (n=1,2,3 routines), default task card — run with `memory="Convo"` (`memory_k=5` for conversation memory, or add `summary_k=10` for summary memory)
- `pal50.json` - Paired-associate learning task (50 word pairs, study + test phases)
- `vviq16.json` - Vividness of Visual Imagery Questionnaire (16 items)
- `rm_singleturn_demo.json` - Relational-memory task, single-turn trial
- `rm_trialchain_demo.json` - Relational-memory task, chained trials
- `rm_dynamic_demo.json` - Relational-memory task, dynamic encoding/test trials
- `rm_episodic_demo.json` - Relational-memory task, episodic conversation (no feedback)
- `rm_episodic_fb_demo.json` - Relational-memory task, episodic conversation with feedback

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
`list_task_library()` pick it up immediately, no registration step. See the
[Task Library guide](../../docs/guides/task_library.md) for the fork-and-PR
workflow to contribute it back.
