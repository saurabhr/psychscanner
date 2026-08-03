# Example Task Definitions

Sample task JSON files for PsychScanner. Fetch any of them by name with
`psychscanner.task_library("name")` — see [`../demonstrations/`](../demonstrations/)
if you want to share a task card of your own instead of adding it here.

## Files

- `example_survey.json` - Simple survey template
- `example_cognitive.json` - Cognitive task template

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
