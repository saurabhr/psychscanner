# RA Guide: Running psychscanner and psychscanner-primal

For a research assistant or co-author who needs to run a `psychscanner`/
`psychscanner-primal` task — on a laptop, on HiPerGator, through UF's
Navigator LLM gateway, or against a hosted API. Read top to bottom the first
time; after that, jump to the section you need.

Official HiPerGator training index (SSO required for some pages):
<https://docs.rc.ufl.edu/training/HiPerGator_training/> — covers Slurm
submission scripts, Slurm MPI scripts, Open OnDemand, data transfer
(Globus/SFTP/rsync/rclone), Jupyter/conda, and a self-enroll Git/GitHub
Canvas course. This guide fills the gaps that index doesn't cover for our
use case: Ollama inside a Slurm job, Navigator, and hosted-API usage.

---

## 0. Before you touch a cluster or an API key: run it locally first

Never debug a task card or a model config for the first time inside a Slurm
job or against a metered API — the feedback loop is minutes-to-hours instead
of seconds, and mistakes cost either allocation or money. Get it working on
your own machine against `mock-llm` (no API key, no network, no GPU) first,
then swap in a real model, then move to the cluster/hosted provider.

```bash
uv venv psyscan --python 3.11
source psyscan/bin/activate
uv pip install -e .        # from inside the psychscanner or psychscanner-primal checkout
```

## 1. Running a task on `psychscanner-primal`

`psychscanner-primal` is the slim distribution — only the feedback-scored
tasks (Reality Monitoring, paired-associate learning), no surveys, no
LangGraph agents, no interpretability backends. See its
[README](https://github.com/saurabhr/psychscanner-primal) for the full
included/excluded list.

```python
from pathlib import Path
from psychscanner import ExpCardInit, ExpCard, ScannerModel, task_library, to_csv

task_path = task_library("rm_singleturn_demo", format="path", dirs="examples/tasks")

card = ExpCardInit(
    model       = "mock-llm",       # swap for a real model once this runs clean
    family      = "mock-llm",
    projectname = "my_first_run",
    proj_dir    = Path.cwd() / "results",
    cogtype     = "no",
    nsim        = 1,
    memory      = "SingleTurn",
    task_file   = task_path,
)

scanner = ScannerModel(expcard=ExpCard(card))
results = scanner.run()
df = to_csv(scanner, path=card.proj_dir)
```

This exact snippet was run for real (not just as documentation) against
`smollm2:360m-instruct-fp16` via Ollama during this guide's own verification
— 8/8 trials, clean CSV, no errors. Swap `model`/`family` per §4 below for
whichever of the four "where to run" paths you're using.

## 2. Running a demonstration_suite module

Each module under
[`examples/demonstration_suite/`](../../examples/demonstration_suite/) is a
standalone `run.py` (simulate) + `analyze.py` (post-process/figures) pair,
reading/writing its own `raw/`, `processed/`, `figures/`, `analysis/`
subfolders. Pick one from the suite's own
[reading-order readme](../../examples/demonstration_suite/readme.md) —
`01_reward_task` is the simplest starting point.

```bash
git clone https://github.com/saurabhr/psychscanner.git
cd psychscanner
uv venv psyscan --python 3.11 && source psyscan/bin/activate
uv pip install -e .

python examples/demonstration_suite/01_reward_task/run.py
python examples/demonstration_suite/01_reward_task/analysis/analyze.py
```

Read the module's own `run.py` docstring first — it states the exact model,
provider, and expected output files (mirrors the header convention used in
[`exp1_1_vviq16/run.py`](../../examples/demonstration_suite/03_personality_survey/advanced/exp1_1_vviq16/run.py)).
Some core modules (`01`–`04`) call hosted APIs directly (Groq, OpenRouter) —
check the docstring before assuming Ollama is enough.

---

## 3. Where to run: four paths

All four paths produce the same `ExpCardInit(model=..., family=..., parameters=...)`
call — only these three arguments change. Everything else about a task card
is identical regardless of where it executes.

### 3.1 Local, self-hosted (Ollama)

The default for development and for co-authors without cluster access.
Install Ollama (<https://ollama.com>), then:

```bash
ollama pull smollm2:360m-instruct-fp16   # smallest model this repo uses; fast on CPU
```

```python
card.model, card.family = "smollm2:360m-instruct-fp16", "ollama"
```

No `base_url` needed — defaults to `http://localhost:11434`. See
[`docs/configuration.md`](../configuration.md) for pointing at a *remote*
Ollama server instead (`parameters={"base_url": ..., "api_key": ...}`) —
the same pattern used for Navigator in §3.3.

**Picking a model size:** this repo's own experiments (`exp1_1_vviq16`, the
HiPerGator job scripts in §4) standardize on `smollm2:360m-instruct-fp16`
for quick iteration — it's fast enough to smoke-test a new task card in
under two minutes. `gemma3:1b-it-qat` and `gemma2:2b` are reasonable next
steps up if a task needs more capability than 360M can reliably follow;
`gemma3:12b-it-qat` is large enough that you should move to §3.2 or §3.4
rather than run it on a laptop CPU. None of these are vision-capable —
multimodal tasks (`04_vlm_task`) need a hosted VLM (§3.4).

### 3.2 HiPerGator (Slurm + Ollama)

For runs too large for a laptop: many models, many conditions, or models
too big to serve locally.

- **Login**: `ssh <gatorlink>@hpg.rc.ufl.edu`. The login node is for editing
  files, submitting jobs, and light work only — a July 2026 policy caps login
  node usage at 4 cores / 32GB; anything heavier goes through Slurm or an
  interactive `srun` session.
- **Storage**: your PI group has a `/blue/<group>/<gatorlink>/` allocation —
  this is where checkouts, venvs, model weights, and job I/O should live, not
  your home directory (small quota, not built for job I/O).
- **Modules**: HPC software is versioned via `module` — `module spider ollama`
  to find available versions, `module load ollama/<version>` to load one.
  `module purge` at the top of every job script to start from a clean
  environment.
- **GPUs**: request with `#SBATCH --gpus=N` and a GPU partition (e.g.
  `hpg-b200`); check current partition names/availability with
  `sinfo -p hpg-b200` since partition names change as hardware rotates in.

**Submission script anatomy** — real, working example, based on
[`examples/hpc_slurm/job_rm_gemma.sh.example`](../../examples/hpc_slurm/job_rm_gemma.sh.example)
(adapted from a script actually used to run reality-monitoring simulations
against Ollama models on HiPerGator; account/paths genericized):

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --account=<your_pi_group>       # e.g. bodegaard — ask your PI
#SBATCH --mem=100gb
#SBATCH --partition=hpg-b200            # check `sinfo` for current GPU partitions
#SBATCH --gpus=1
#SBATCH --cpus-per-task=20
#SBATCH --time=50:00:00                 # HH:MM:SS wall-clock limit — jobs are
                                         # killed hard when this expires
#SBATCH --output=out_%j.log
#SBATCH --error=err_%j.log

module purge
module load ollama/0.9.0                # check `module spider ollama` for current version
module load conda

cd /blue/<your_pi_group>/<gatorlink>/psychscanner
source .venv/bin/activate

# Serve Ollama in the background, then run your Python driver against it —
# see the Ollama-on-HiPerGator steps below.
```

**Commands you'll actually use:**

| Command | Purpose |
|---|---|
| `sbatch job.sh` | Submit a job script |
| `squeue -u $USER` | See your queued/running jobs |
| `scancel <jobid>` | Kill a job |
| `seff <jobid>` | Post-run efficiency report (CPU/mem/GPU utilization) — check this after a first run to right-size `--mem`/`--cpus-per-task` for the next |
| `srun --partition=hpg-b200 --gpus=1 --mem=32gb --time=01:00:00 --pty bash` | Interactive session — for debugging a job script before committing to a long `sbatch` run |

Start every new task with a short interactive `srun` session and a tiny
`--time` budget to shake out path/env errors, *then* write the real `sbatch`
script with a realistic wall-clock limit. A job that dies at hour 49 of a
50-hour `--time` because of a typo in a path is the single most common way to
waste a HiPerGator allocation.

**Ollama inside the job** — there's no official HiPerGator page for this;
the pattern below is copied from real, working job scripts in this checkout:
[`job_rm_gemma.sh.example`](../../examples/hpc_slurm/job_rm_gemma.sh.example) and
[`job_rm_llama.sh.example`](../../examples/hpc_slurm/job_rm_llama.sh.example).

1. `module load ollama/<version>` (check `module spider ollama` for what's
   currently installed).
2. Point model storage at your `/blue` allocation, not the default cache
   location (home-directory quota is too small for model weights):
   ```bash
   export OLLAMA_MODELS=/blue/<your_pi_group>/<gatorlink>/ollama_models/
   ```
3. Start the server in the background *inside the job*, then give it a few
   seconds to come up before hitting it:
   ```bash
   ollama serve >> ollama_server.log 2>&1 &
   sleep 10
   ```
4. Pull the model, run your simulation, then `ollama rm <model>` when done
   with it — models are large (multi-GB) and `/blue` quotas are finite:
   ```bash
   ollama pull <model>
   python your_run_script.py --taskjson <task> --modelname <model> --familyname ollama
   ollama rm <model>
   ```
5. Point `ExpCardInit(family="ollama", ...)` at the local server — no
   `base_url` override needed, it's on `localhost:11434` inside the same job.

The two example scripts loop over a models file and a tasks file
(one model/task name per line), pulling → running → removing each model in
turn, so a single job can sweep many model×task combinations without holding
every model on disk at once. Adapt the loop, don't copy the exact filenames.

### 3.3 UF Navigator (AIBHS)

Navigator is a hosted LLM gateway used as an alternative to self-hosting via
Ollama, run through AIBHS ("Artificial Intelligence in Biomedical and Health
Sciences," a UF College of Medicine program under the Intelligent Clinical
Care Center). **This section is a placeholder** — the public AIBHS site has
no technical documentation, and I don't have Navigator's endpoint, auth
mechanism, or model names confirmed yet. Once you have those (from AIBHS
IT/program staff), they belong in a **private, non-public note** — never in
this file, since it's a shared repo doc. The team's working copy of that note
lives outside this repo entirely, at `PSYCHSCANNER/private_notes/
aibhs_navigator_provider.md` (not committed, not synced to GitHub).

The expected shape, if Navigator turns out to be OpenAI-API-compatible
(true for most hosted LLM gateways), follows the same pattern as a remote
Ollama server in [`docs/configuration.md`](../configuration.md):

```python
card.model, card.family = "<navigator-model-name>", "openai"   # unconfirmed
card.parameters = {
    "base_url": "<navigator-endpoint>",   # from AIBHS, not committed here
    "api_key": "<from env, never hardcoded>",
}
```

If Navigator isn't OpenAI-compatible, it needs a real LangChain integration
package or a small custom `BaseChatModel` wrapper before `family` can point
at it. Update this section once confirmed.

### 3.4 Online / hosted APIs

For models bigger than what's practical to self-host, or for tasks needing a
specific provider's model (e.g. GPT, Claude, Gemini). Full table:
[`docs/installation.md#supported-providers`](../installation.md#supported-providers).
Only `ollama` ships with no extra install; every other family needs its own
LangChain package, e.g.:

```bash
uv pip install langchain-openai      # for family="openai"
uv pip install langchain-anthropic   # for family="anthropic"
```

```python
import os
os.environ["OPENAI_API_KEY"] = "..."   # or export in shell / use python-dotenv

card.model, card.family = "gpt-4o-mini", "openai"
```

Two of the core `examples/demonstration_suite/` modules already do this for
real (`03_personality_survey` via Groq, `04_vlm_task` via OpenRouter) — read
their `run.py` for a working reference rather than starting from scratch.
**Cost note:** hosted APIs are metered — always smoke-test with `mock-llm`
or a cheap/free-tier model (§3.1, or OpenRouter's `:free` model suffixes)
before pointing `nsim`/a factorial design at a paid model.

---

## 4. Globus (moving data in and out)

For anything past a few files, use Globus rather than `scp`/`rsync` over the
login node (more reliable for large transfers, resumable, doesn't tie up your
terminal). HiPerGator's endpoint and setup steps are on the official page:
Data Transfer Tools section of
<https://docs.rc.ufl.edu/training/HiPerGator_training/>. Rough shape:

1. Set up a Globus account at <https://www.globus.org> (UF SSO works).
2. Activate the HiPerGator endpoint (search "University of Florida" in the
   Globus web app's endpoint search, or use the CLI: `globus endpoint search
   "University of Florida"`).
3. Activate a second endpoint for the other side — your laptop (install
   Globus Connect Personal) or another institution's HiPerGator-side share.
4. Transfer via the web app, or the CLI once both endpoints are activated:
   ```bash
   globus transfer <hpg-endpoint-id>:/blue/<group>/<user>/psychscanner/results \
                    <local-endpoint-id>:/path/to/local/results --recursive
   ```

Use this to pull `raw/`/`processed/`/`figures/` output folders back from a
demonstration_suite run, or to push a large persona/stimulus dataset onto
`/blue` before a job needs it.

---

## 5. Custom / boutique experiments: combining task-card options

The bundled demos each fix one combination of these options. A new,
one-off ("boutique") experiment is usually just a different combination of
the same `ExpCardInit` fields — no framework changes needed. The
independent axes, each documented in more depth in the linked guide:

| Axis | Field(s) | Values | Guide |
|---|---|---|---|
| Provider | `model`, `family`, `parameters` | any of §3.1–§3.4 | [installation.md](../installation.md) |
| Memory / chain structure | `memory`, `chain_type` | `memory`: `SingleTurn` \| `Convo`; `chain_type`: `item` (single-turn) \| `trial` (trial-chain) \| `task` (episodic-chain) | [memory_types.md](memory_types.md) |
| Context window | `memory_k`, `summary_k` | int, or `None` for unbounded | [memory_types.md](memory_types.md) |
| Persona conditioning | `cogtype`, `persona_files` | `cogtype`: `"no"` \| `"assistant"` \| `"custom"`; `persona_files`: list of persona JSONs, only used when `cogtype="custom"` | [cognitive_tasks.md](cognitive_tasks.md) |
| Task content | `task_file` | dict, or a path resolved via `task_library()` | [task_library.md](task_library.md) |
| Response parsing | `parser`, `parser_raw`, `parser_config`, `trial_parsers` | registered name, Pydantic model, or callable | [custom_parsers.md](custom_parsers.md) |
| Trial-level feedback | `feedback`, `feedback_fn` | bool + a `FeedbackBase` subclass | [../examples/feedback_loop.md](../examples/feedback_loop.md) |
| Conditional branching | `next_trial`, `next_trial_fn` | bool + a `NextTrialBase` subclass | [../examples/conditional_next_trial.md](../examples/conditional_next_trial.md) |
| Multimodal / tool use | `tools` | list of LangChain tools | [12_tool_binding_and_multimodal.ipynb](../../examples/demonstration_suite/00_install_from_github.ipynb) tutorial series |
| Session recovery | `tunnel_status`, `tunnel_k` | `"0"`/`"1"`, checkpoint interval | [session_recovery.md](session_recovery.md) |
| Participants | `nsim` | int, simulated participants per condition | — |

**Recipe — a boutique 2×3 factorial** (persona × memory), the pattern every
`advanced/exp*` module in this repo follows: loop over the Cartesian product
of your factor levels, constructing one `ExpCardInit` per cell, same
`task_file` throughout:

```python
CONDITIONS = [
    {"memory": "SingleTurn", "chain_type": "item",  "label": "item_ST"},
    {"memory": "Convo",      "chain_type": "task",  "label": "task_Cv"},
    {"memory": "Convo",      "chain_type": "trial", "label": "trial_Cv"},
]
PERSONAS = {"weird": "personas/weird_axis.json", "non_weird": "personas/non_weird_axis.json"}

for persona_name, persona_file in PERSONAS.items():
    for cond in CONDITIONS:
        card = ExpCardInit(
            model=MODEL, family=FAMILY,
            cogtype="custom", persona_files=[persona_file],
            task_file=TASK_FILE, memory=cond["memory"], chain_type=cond["chain_type"],
            projectname=f"boutique_{persona_name}_{cond['label']}",
            proj_dir=RESULTS_DIR, tunnel_status="0",
        )
        ScannerModel(expcard=ExpCard(card)).run()
```

This is exactly [`exp1_1_vviq16/run.py`](../../examples/demonstration_suite/03_personality_survey/advanced/exp1_1_vviq16/run.py)'s
structure, generalized — read that file for the complete, working version
including combining results across cells and generating figures. Adding a
third axis (e.g. feedback on/off) is the same pattern with one more nested
loop; adding conditional branching (`next_trial`) or custom feedback
(`feedback_fn`) means writing a small subclass once and passing it into
every cell's `ExpCardInit` unchanged.

---

## 6. Common pitfalls

- **Job dies immediately, log is empty** — almost always a `module load`
  typo or a path that doesn't exist on the compute node (compute nodes don't
  share your login-node shell state beyond what the job script itself does).
  Reproduce with a short interactive `srun` first (§3.2).
- **`OLLAMA_MODELS` not set before `ollama serve`** — the server will default
  to your home directory, hit the quota, and fail silently mid-pull. Set the
  env var *before* starting the server, not after.
- **Job killed at the `--time` limit mid-run** — check `seff <jobid>` after
  a first attempt and pad the wall-clock estimate; a killed job doesn't
  leave partial CSVs in a clean state to resume from.
- **Task/model name mismatch** — see
  [`guides/task_library.md`](task_library.md) for why a task card's internal
  `"taskname"` field must match its filename; a bundled task
  simulated a dozen times under the wrong name is a wasted allocation, not
  just a wasted afternoon.
- **Storage: home directory vs `/blue`** — model weights, venvs, and job
  output all belong on `/blue/<group>/<user>/`, never `$HOME` (small quota,
  not tuned for job I/O).
- **Small local models degrade on structured prompts** — `smollm2:360m` and
  similar tiny models can drift off-task (echo the system prompt, answer a
  different question than asked) rather than erroring out. That's a model
  capability limit, not a pipeline bug — check the raw response column
  before assuming something's broken. This is itself a documented finding
  in the project's papers (`papers/results_index.md`), not just a footgun.
