# Docs archive

Pages pulled out of the published site's nav (`mkdocs.yml`) on 2026-08-19,
kept here for reference instead of deleted. None of these files are built
into the site (`docs/archive/` is not in `mkdocs.yml` nav).

## examples_reality_monitoring.md

Was `docs/examples/reality_monitoring.md`, linked from the Examples nav as
"Reality Monitoring". Walked through the RM (reality monitoring) paradigm —
judging whether a word was given (perceived) or self-generated (imagined) —
using `tasks/rm_task.json`, `Response_part_1_rm`/`Response_part_2_rm`, and a
`memory="Convo"` run. It worked: this was a real, runnable example matching
`05_rm_task.ipynb`.

Removed because no reality-monitoring task exists in
`examples/demonstration_suite/` (checked — the `rm_` filename prefix used
elsewhere in the repo is this same reality-monitoring paradigm, not a
demonstration-suite entry). The Examples page now points at the
demonstration-suite write-ups instead. The underlying task JSON, parsers, and
`guides/cognitive_tasks.md`'s RM schema walkthrough are untouched — this only
removes the narrative example page.

## tutorials_05_rm_task.ipynb / tutorials_07_rm_feedback_task.ipynb

Were `docs/tutorials/05_rm_task.ipynb` and `docs/tutorials/07_rm_feedback_task.ipynb`,
rendered into the site via the `mkdocs-jupyter` plugin. Ran the same RM task
end to end (05) and with trial-by-trial feedback injected (07); both executed
successfully as tutorials. Removed from the nav for the same reason as the
example page above — no RM demonstration in `demonstration_suite/`. The
source notebooks in `examples/05_rm_task.ipynb` and
`examples/07_rm_feedback_task.ipynb` are untouched; only the doc-site copies
and their `mkdocs.yml` nav entries were removed.

## examples_basic_survey.md

Was `docs/examples/basic_survey.md`, "Simple Survey" in the Examples nav. Ran
the built-in VVIQ-16 questionnaire with a minimal `ExpCardInit` call
(`DefaultLiteralVivid15` parser, `SingleTurn` memory, no cogtype). It worked
as a minimal-example page.

Replaced because `examples/demonstration_suite/03_personality_survey/`
demonstrates this same VVIQ-16 survey as one of its 8 cells
(`advanced/exp1_1_vviq16`, marked complete and reported in the manuscript),
plus 3 more memory conditions and 2 persona conditions the old page didn't
cover. The new `docs/examples/demonstration_suite/03_personality_survey.md`
supersedes it.

## examples_multi_persona.md

Was `docs/examples/multi_persona.md`, "Multi-Persona Study" in the Examples
nav, and separately linked from the User Guide as "Running a Survey with
Persona Levels". Explained the `persona_statements` JSON format and running
one card per persona. It worked, but was a mechanism walkthrough with no real
run data attached.

Replaced because `03_personality_survey` runs real persona-conditioned
studies (`weird` vs `non_weird_jung` persona files, 2 persona x 4 memory
cells) with actual output data and a report. The User Guide nav entry
("Running a Survey with Persona Levels") now points at the new
`docs/examples/demonstration_suite/03_personality_survey.md` instead of this
file.

## examples_feedback_loop.md

Was `docs/examples/feedback_loop.md`, "Feedback Loop System" in the Examples
nav. Documented the `FeedbackBase` interface (`on_response`, `inject_feedback`)
and matched notebooks `06_feedback_api.ipynb`/`07_rm_feedback_task.ipynb`. It
worked as an API-reference-style walkthrough of the feedback mechanism, with
no full study behind it.

Replaced because `01_reward_task` (3-armed bandit, `FeedbackBase`-driven
reward learning, 3 agent types, real Groq calls) and `02_association_memory`
(single-turn / trial-chain / episodic-chain, correct/incorrect feedback,
reward feedback) both demonstrate the same `FeedbackBase` mechanism with real
run data and reports. New pages:
`docs/examples/demonstration_suite/01_reward_task.md` and
`docs/examples/demonstration_suite/02_association_memory.md`.

## Not archived

`docs/examples/conditional_next_trial.md` — no equivalent exists in
`demonstration_suite/` (grepped for `next_trial` usage there; none found), so
it stays as the only documentation of that feature.

`guides/cognitive_tasks.md`, `guides/memory_types.md`,
`guides/custom_parsers.md`, `blog/understanding_task_cards.md`,
`api/parsers.md`, `cli.md`, `index.md` — these reference RM only as a schema
illustration inside broader guides, not as a featured standalone example.
Left as-is per explicit scope: only the featured example page and tutorial
notebooks were in scope for removal, not every schema example that happens to
use an RM-shaped task JSON.
