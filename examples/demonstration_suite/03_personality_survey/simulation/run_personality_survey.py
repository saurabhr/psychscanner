"""Demonstration 03 — Personality Survey (BFI-44, subsampled).

Runs a subset of the Big Five Inventory (3 items x 5 OCEAN traits = 15
items) as a psychscanner survey task, crossing 2 persona conditions with 4
memory conditions, to show off the four features listed in ../readme.md:

  1. Two levels of persona            -> PERSONAS (weird / non_weird_jung)
  2. Independent (stateless) sampling -> "stateless" condition
  3. Conversation & Summary memory    -> "conversation" / "summary_windowed"
  4. Windowing Convo & Summary memory -> "conversation_windowed" / "summary_windowed"

Model calls are real (Groq, family="groq"). Requires GROQ_API_KEY in .env.
Groq's free-tier TPM cap is hit routinely by Convo conditions (context grows
every trial); psychscanner's model-call wrapper already retries 429s with
exponential backoff (tenacity, up to 8 attempts — see
src/psychscanner/memories/single_turn_convo.py::_invoke_with_retry), which is
what we rely on here rather than adding a second retry layer on top.

Run from the psychscanner repo root:
    source .venv/bin/activate
    python examples/demonstration_suite/03_personality_survey/simulation/run_personality_survey.py

Design
------
Full BFI-44 is 44 items; kept lean here to 3 items/trait (15 total) so the
whole 2 persona x 4 memory grid stays in the low hundreds of model calls
instead of the ~1800 a full-instrument run would need.

Persona (2): weird, non_weird_jung  (axis file x 2 of the 5 traits_n5.json
statements -> 2 sims each; a 5-statement run is possible by using the full
traits_n5.json but was cut for call budget)
Memory (4):
  stateless               chain_type=item  memory=SingleTurn                 (feature 2)
  conversation             chain_type=task  memory=Convo  memory_k=-1 sk=0    (feature 3, unbounded)
  conversation_windowed    chain_type=task  memory=Convo  memory_k=4  sk=0    (feature 4, hard window)
  summary_windowed         chain_type=task  memory=Convo  memory_k=4  sk=2    (features 3+4, window+summary)

memory_k=4/summary_k=2 mirrors examples/11_memory_context_management.ipynb's
worked example: once the window fills, each trial's overflow settles at
exactly 2 messages (1 dropped human + 1 dropped AI turn), so summary_k must
be <=2 to ever actually fire — summary_k=4 (tried first) silently never
summarized anything, confirmed by inspecting `pred_dict["summary"]` trial by
trial in a dry run.

Participants per persona x condition: 2
Total: 2 personas x 4 conditions x 2 sims x 15 items = 240 trial rows.

Outputs
-------
data/raw/{persona}_{cond}.csv      -- 30 rows each (2 sims x 15 items)
data/processed/combined.csv        -- 240 rows, all conditions
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from psychscanner import ExpCard, ExpCardInit, ScannerModel, to_csv

# ── paths ─────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
DEMO_ROOT = HERE.parent                      # examples/demonstration_suite/03_personality_survey/
REPO_ROOT = DEMO_ROOT.parents[2]             # psychscanner/

load_dotenv(REPO_ROOT / ".env")

PERSONA_DIR = DEMO_ROOT / "advanced" / "personas"
RAW_DIR     = DEMO_ROOT / "data" / "raw"
PROC_DIR    = DEMO_ROOT / "data" / "processed"

for d in (RAW_DIR, PROC_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── build a 15-item BFI-44 subset (3 items/trait, first N in original order) ─
ITEMS_PER_TRAIT = 3
full_task = json.loads((REPO_ROOT / "examples" / "tasks" / "bfi44.json").read_text())
kept: dict = {}
counts = {t: 0 for t in full_task["contexts_id"]}
for trcode, item in full_task["items"].items():
    trait = trcode[0]
    if counts.get(trait, ITEMS_PER_TRAIT) < ITEMS_PER_TRAIT:
        kept[trcode] = item
        counts[trait] += 1
task_subset = {**full_task, "items": kept}
TASK_FILE = HERE / "_bfi44_subset.json"
TASK_FILE.write_text(json.dumps(task_subset, indent=2))
print(f"BFI-44 subset: {len(kept)} items ({ITEMS_PER_TRAIT}/trait) -> {TASK_FILE.name}")

# ── trim traits_n5.json (5 statements) down to 2 -> 2 sims/persona ──────────
# ponytail: nsim=2 (not the 5 in examples/demonstration_suite/03_personality_survey/advanced) to fit
# Groq's free-tier daily token cap -- llama-3.3-70b-versatile burned its
# entire 100k TPD budget partway through a dry run, hence the model switch
# below too. Bump back to the full traits_n5.json for a richer n once on a
# paid tier / after quota resets.
traits_full = json.loads((PERSONA_DIR / "traits_n5.json").read_text())
TRAITS_N2 = HERE / "_traits_n2.json"
TRAITS_N2.write_text(json.dumps({"persona_statements": traits_full["persona_statements"][:2]}, indent=2))

# ── experiment parameters ────────────────────────────────────────────────
# llama-3.3-70b-versatile hit Groq's 100k-token/day cap mid-run (see note
# above) -- switched to gpt-oss-120b, which draws from a separate per-model
# quota and was verified stable (json_schema structured output) on short
# Convo chains during dry runs; it did intermittently emit unparseable
# output on a full 44-item chain, which is exactly why this demo keeps the
# survey subsampled rather than the full instrument.
MODEL  = "openai/gpt-oss-120b"
FAMILY = "groq"
PARAMS = {"temperature": 0}

CONDITIONS = [
    {"label": "stateless",             "memory": "SingleTurn", "chain_type": "item", "memory_k": -1, "summary_k": 0},
    {"label": "conversation",          "memory": "Convo",      "chain_type": "task", "memory_k": -1, "summary_k": 0},
    {"label": "conversation_windowed", "memory": "Convo",      "chain_type": "task", "memory_k": 4,  "summary_k": 0},
    {"label": "summary_windowed",      "memory": "Convo",      "chain_type": "task", "memory_k": 4,  "summary_k": 2},
]

PERSONAS = {
    "weird":          [PERSONA_DIR / "weird" / "weird_axis.json", TRAITS_N2],
    "non_weird_jung": [PERSONA_DIR / "non_weird" / "non_weird_axis.json",
                        PERSONA_DIR / "jung" / "jung_axis.json",
                        TRAITS_N2],
}

# ── run simulations ──────────────────────────────────────────────────────
frames: list[pl.DataFrame] = []

for persona_name, persona_files in PERSONAS.items():
    for cond in CONDITIONS:
        run_label = f"{persona_name}_{cond['label']}"
        print(f"\n{'=' * 60}\n  {run_label}\n{'=' * 60}")

        csv_path = RAW_DIR / f"{run_label}.csv"
        if csv_path.exists():
            df = pl.read_csv(csv_path)
            frames.append(df)
            print(f"  skipped (exists): {csv_path.name}  ({len(df)} rows)")
            continue

        session_dir = RAW_DIR / run_label
        if session_dir.exists():
            shutil.rmtree(session_dir)

        card = ExpCardInit(
            model         = MODEL,
            family        = FAMILY,
            parameters    = PARAMS,
            cogtype       = "custom",
            persona_files = persona_files,
            task_file     = TASK_FILE,
            parser        = "1",                 # -> DefaultLiteralAgree from task JSON
            parser_config = {"method": "json_schema"},
            memory        = cond["memory"],
            chain_type    = cond["chain_type"],
            summary_k     = cond["summary_k"],
            memory_k      = cond["memory_k"],
            projectname   = run_label,
            proj_dir      = RAW_DIR,
            tunnel_status = "0",
        )
        # gpt-oss-120b occasionally emits an unparseable structured-output
        # response (see script docstring); one retry from scratch is enough
        # in practice, and a failing cell is skipped rather than aborting
        # the whole run -- ponytail: no backoff/queue here, just try twice.
        try:
            expcard = ExpCard(card)
            scanner = ScannerModel(expcard=expcard)
            scanner.run()
        except Exception as exc:
            print(f"  FAILED once ({exc}); retrying {run_label}...")
            if session_dir.exists():
                shutil.rmtree(session_dir)
            try:
                expcard = ExpCard(card)
                scanner = ScannerModel(expcard=expcard)
                scanner.run()
            except Exception as exc2:
                print(f"  SKIPPING {run_label} after 2 failures: {exc2}")
                continue

        df = to_csv(scanner, path=csv_path)
        df = df.with_columns([
            pl.lit(persona_name).alias("persona"),
            pl.lit(cond["label"]).alias("memory_cond"),
            pl.lit(cond["memory_k"]).alias("memory_k"),
            pl.lit(cond["summary_k"]).alias("summary_k"),
        ])
        df.write_csv(csv_path)
        frames.append(df)
        print(f"  saved {len(df)} rows -> {csv_path.name}")

combined = pl.concat(frames, how="diagonal")
combined_path = PROC_DIR / "combined.csv"
combined.write_csv(combined_path)
print(f"\nCombined: {combined.shape} -> {combined_path}")
