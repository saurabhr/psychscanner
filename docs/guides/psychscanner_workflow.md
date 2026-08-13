# PsychScanner Workflow

How a task card and a persona become a running session. The `ExpCard` is
the **experiment card** — the full specification of one experimental
session (model, task, memory architecture, persona conditioning), the same
role a pre-registration or session protocol plays in a human study. This
page traces what happens to it end to end, from your inputs to the trial
records written to disk.

## Slide 1 — High-level flow

- `ExpCard` is the central configuration object.
- It receives all external settings and drives the scanner lifecycle.
- The runtime chain is:
  1. Inputs → `ExpCard`
  2. `ExpCard` → `ScannerModel`
  3. `ScannerModel` → `TaskRunner`
  4. `TaskRunner` → outputs

```mermaid
graph LR
    classDef t1 fill:#E8F4F8,stroke:#1B4F72,stroke-width:2px,color:#1B4F72
    classDef t2 fill:#D5F5E3,stroke:#196F3D,stroke-width:2px,color:#196F3D
    classDef t3 fill:#FCF3CF,stroke:#9A7D0A,stroke-width:2px,color:#9A7D0A
    classDef t4 fill:#FADBD8,stroke:#7B241C,stroke-width:2px,color:#7B241C
    classDef t5 fill:#EBDEF0,stroke:#6C3483,stroke-width:2px,color:#6C3483
    classDef psyscan fill:#FFF3CD,stroke:#FF9800,stroke-width:3px,color:#E65100

    A["Inputs"]:::t1 --> B["ExpCard"]:::t1
    B --> C["ScannerModel"]:::t2
    C --> D["scanner_data<br/>generator"]:::t3
    D --> E["TaskRunner"]:::t4
    E <-->|apply feedback| FB["feedback_function"]:::t1
    E --> F["SimulationModel"]:::t5
    F --> G[".psyscan files"]:::psyscan
```

## Slide 2 — Inputs and staging

- Inputs:
  - `task_file.json`
  - project config
  - `persona_file_path(s)`
  - parser settings
  - model parameters
  - memory settings
  - feedback settings
  - `feedback_function`
- `ExpCard` responsibilities:
  - validate and normalize inputs
  - load task and persona data
  - configure parser settings
  - create output directories
  - construct `SessionTunnel`

## Slide 2.5 — Detailed grid view (Horizontal Flow)

```mermaid
graph LR
    classDef t1 fill:#E8F4F8,stroke:#1B4F72,stroke-width:2px,color:#1B4F72
    classDef t2 fill:#D5F5E3,stroke:#196F3D,stroke-width:2px,color:#196F3D
    classDef t3 fill:#FCF3CF,stroke:#9A7D0A,stroke-width:2px,color:#9A7D0A
    classDef t4 fill:#FADBD8,stroke:#7B241C,stroke-width:2px,color:#7B241C
    classDef t5 fill:#EBDEF0,stroke:#6C3483,stroke-width:2px,color:#6C3483
    classDef psyscan fill:#FFF3CD,stroke:#FF9800,stroke-width:3px,color:#E65100

    subgraph COL1["Inputs"]
        direction TB
        P1["task_file.json"]:::t1
        P2["project config"]:::t1
        P3["persona_file_path(s)"]:::t1
        P4["parser settings"]:::t1
        P5["model parameters"]:::t1
        P6["memory settings"]:::t1
        P7["feedback settings"]:::t1
        P8["feedback_function"]:::t1
    end

    subgraph COL2["Staging"]
        direction TB
        S1["ExpCard"]:::t2
        S2["SessionTunnel"]:::t2
        S3["TaskData"]:::t2
    end

    subgraph COL3["Runtime"]
        direction TB
        R1["ScannerModel"]:::t3
        R2["PromptGen"]:::t3
        R3["AgentInit"]:::t3
        R4["TaskRunner"]:::t4
        R5["TrialLoop"]:::t4
        R6["feedback_function"]:::t1
    end

    subgraph COL4["Outputs"]
        direction TB
        O1["system_prompts"]:::t4
        O2["task_prompts"]:::t4
        O3["SimulationModel"]:::t5
        O4[".psyscan files"]:::psyscan
    end

    P1 --> S1
    P2 --> S1
    P3 --> S1
    P4 --> S1
    P5 --> S1
    P6 --> S1
    P7 --> S1
    P8 --> S1

    S1 --> S2
    S1 --> S3
    S1 --> R1

    R1 --> R2
    R2 --> O1
    R2 --> O2
    R1 --> R3
    R3 --> R4

    O1 -->|loop| R4
    O2 --> R4
    R4 --> R5
    R5 -->|model response| R6
    R6 -->|eval/update| R5
    R5 --> O3
    O3 --> O4

    R4 -->|checkpoint| S2
    S2 -->|resume| R1
```


- This grid shows horizontal stage columns and vertical detail within each stage.
- `TaskRunner` receives the AI agent and memory chain from `AgentInitializer` and loops over both `system_prompts` and `task_prompts`.

## Slide 2.6 — Grid view with horizontal flow

```mermaid
graph LR
    classDef t1 fill:#E8F4F8,stroke:#1B4F72,stroke-width:2px,color:#1B4F72
    classDef t2 fill:#D5F5E3,stroke:#196F3D,stroke-width:2px,color:#196F3D
    classDef t3 fill:#FCF3CF,stroke:#9A7D0A,stroke-width:2px,color:#9A7D0A
    classDef t4 fill:#FADBD8,stroke:#7B241C,stroke-width:2px,color:#7B241C
    classDef t5 fill:#EBDEF0,stroke:#6C3483,stroke-width:2px,color:#6C3483
    classDef psyscan fill:#FFF3CD,stroke:#FF9800,stroke-width:3px,color:#E65100

    subgraph ROW1["Inputs → Staging → Runtime → Outputs"]
        subgraph INPUTS["Inputs"]
            P1["task_file.json"]:::t1
            P2["project config"]:::t1
            P3["persona_file_path(s)"]:::t1
            P4["parser settings"]:::t1
            P5["model parameters"]:::t1
            P6["memory settings"]:::t1
            P7["feedback settings"]:::t1
            P8["feedback_function"]:::t1
        end

        subgraph STAGING["Staging"]
            S1["ExpCard"]:::t2
            S2["SessionTunnel"]:::t2
            S3["TaskData"]:::t2
        end

        subgraph RUNTIME["Runtime"]
            R1["ScannerModel"]:::t3
            R2["PromptGen"]:::t3
            R3["AgentInit"]:::t3
            R4["TaskRunner"]:::t4
            R5["TrialLoop"]:::t4
            R6["feedback_function"]:::t1
        end

        subgraph OUTPUTS["Outputs"]
            O1["system_prompts"]:::t4
            O2["task_prompts"]:::t4
            O3["SimulationModel"]:::t5
            O4[".psyscan files"]:::psyscan
        end
    end

    P1 --> S1
    P2 --> S1
    P3 --> S1
    P4 --> S1
    P5 --> S1
    P6 --> S1
    P7 --> S1
    P8 --> S1

    S1 --> S2
    S1 --> S3
    S1 --> R1

    R1 --> R2
    R2 --> O1
    R2 --> O2
    R1 --> R3
    R3 --> R4

    O1 -->|loop| R4
    O2 --> R4
    R4 --> R5
    R5 -->|model response| R6
    R6 -->|eval/update| R5
    R5 --> O3
    O3 --> O4

    R4 -->|checkpoint| S2
    S2 -->|resume| R1
```


- This horizontal flow grid shows the complete workflow progression from inputs through staging, runtime, and outputs.
- The layout emphasizes the linear flow of data through the system.

## Slide 3 — Runtime components

- `ScannerModel` turns `ExpCard` into actionable prompt data:
  - `system_prompts`
  - `task_prompts`
- `ScannerModel.agent()` creates the AI agent and memory chain (via `AgentInitializer` + `single_turn_convo_node()`).
- Each `TaskRunner` (one per system prompt) receives:
  - the AI agent with memory chain
  - one system prompt
  - `task_prompts`
- `TaskRunner.execute()` runs the trial loop and records results.

## Slide 4 — Loop behavior

```mermaid
graph LR
    classDef t3 fill:#FCF3CF,stroke:#9A7D0A,stroke-width:2px,color:#9A7D0A
    classDef t4 fill:#FADBD8,stroke:#7B241C,stroke-width:2px,color:#7B241C
    classDef t5 fill:#EBDEF0,stroke:#6C3483,stroke-width:2px,color:#6C3483

    SP["system_prompts"]:::t3
    TP["task_prompts"]:::t3
    AG["AI agent + memory"]:::t3
    TR["TaskRunner"]:::t4
    FB["feedback_function"]:::t1
    OUT["simulation results"]:::t5

    SP -->|loop| TR
    TP --> TR
    AG --> TR
    TR <-->|apply feedback| FB
    TR --> OUT
```

## Slide 5 — Outputs and resume

- `TaskRunner.execute()` returns a list of per-trial result dicts; `ScannerModel.model_dump()` wraps them in `SimulationModel` / `TaskSimulationModel`.
- Outputs are saved as `*.psyscan` files.
- `SessionTunnel` logs checkpoints; `ScannerModel.tunnel_systemtrials()` uses them to compute a resume index.

> When tunnel mode is enabled, the workflow can restart from the last completed system prompt.


## Key stages

1. Input assembly
   - Persona file path(s), task JSON, model parameters, memory type, parser config, feedback config, and project/tunnel settings are provided.
   - All of these external inputs are encoded into `ExpCard` as the single experiment configuration object.
2. Experiment card creation
   - `ExpCard` validates inputs, loads task/persona data, configures parser settings, creates output directories, and constructs `SessionTunnel`.
3. Scanner model preparation
   - `ScannerModel` translates the experiment card into `scanner_data` and `task_prompts`.
4. Agent initialization
   - `ScannerModel.agent()` builds the LLM agent graph for the selected memory chain (`SingleTurn` or `Convo`) via `single_turn_convo_node()`, attaches it to an `AgentInitializer`, and that agent is passed into a `TaskRunner`.
5. Trial execution
   - `ScannerModel.run()` loops over system prompts, creating one `TaskRunner` per system prompt; each `TaskRunner` iterates its own task trials, invokes the agent, applies optional feedback, and collects trial outputs.
6. Output persistence
   - `ScannerModel.model_dump()` wraps the collected trial dicts in `SimulationModel` / `TaskSimulationModel` and writes `.psyscan` files; session checkpoints are logged separately via `SessionTunnel`.
7. Resume support
   - When tunnel mode is enabled, `ScannerModel.tunnel_systemtrials()` reads past logs via `SessionTunnel` and computes the resume index.



## Class Diagrams theme

### Alternative Workflow View - Class Diagram Style

```mermaid
classDiagram
    direction LR

    class ExpCard {
        +card_in: ExpCardInit
        +task_data: dict
        +persona_data: dict
        +data_root_dir: Path
        +session_tunnel: SessionTunnel
    }

    class ScannerModel {
        +expcard: ExpCard
        +tunnel: SessionTunnel
        +tunnel_systemtrials()
        +agent(agent_cfg, memory)
        +run(progress_bar)
        +model_dump()
    }

    class TaskRunner {
        +system_message: str
        +tasktrials: dict
        +execute()
    }

    class SimulationModel {
        +simdata: List~TrialSimulationModel~
    }

    class SessionTunnel {
        +create_tunnel()
        +scan_checkpoint()
        +subscan_checkpoint()
        +end_checkpoint()
        +load_tunnel_logs()
    }

    ExpCard --> ScannerModel : configures
    ExpCard --> SessionTunnel : constructs
    ScannerModel --> TaskRunner : one instance per system prompt
    TaskRunner --> SimulationModel : trial result dicts
    ScannerModel --> SimulationModel : wraps and writes via model_dump
    ScannerModel --> SessionTunnel : checkpoints per system prompt
```

Note: `ExpCard` and `TaskRunner` do the work described above inline in `__init__`/`execute` — they don't expose separate `validate_inputs()`/`run_trials()`-style methods. The method names above are the real public methods on each class.



Based on the architecture of PsychScanner compared to the Inspect framework from the UK AI Security Institute (AISI), there are several fundamental differences in their purpose, target domain, and core abstractions.

While both are frameworks that run Language Models through a pipeline of tasks, Inspect is a generalized machine learning benchmarking tool, whereas PsychScanner is a domain-specific framework tailored for simulating psychological experiments and cognitive behavioral paradigms.

Here is a breakdown of how they differ:

1. Core Focus & Domain
Inspect: Built to evaluate generalized capabilities and safety of large language models. It targets benchmarks like coding tasks, advanced reasoning, and agentic problem solving. It provides out-of-the-box support for taking standard AI benchmarks (like MMLU or SWE-bench) and evaluating models against them.
PsychScanner: Built explicitly to simulate human psychological behaviors across surveys and cognitive tasks. Its focus is on evaluating how an LLM acts when prompted with human-like personas and subjective items (like taking the VVIQ survey or ranking emotional relatedness).
2. Architecture & Abstractions
Inspect Models: Uses generic concepts like Datasets, Solvers (prompts/strategies applied to the model), Scorers (evaluating accuracy against a known target), and Tasks.
PsychScanner Models: Uses psychology experiment-based semantics. It relies on an ExpCard to configure experimental conditions, defines Persona data to assign subjective human traits to the model, and runs a TaskRunner (simulating experimental trials).
3. Memory & Trial Continuity
Inspect: Primarily evaluates interactions based on agent behavior and tool usages (like bash, Python, browser manipulation).
PsychScanner: Deeply focuses on stateful psychological persistence. It utilizes memory settings (`memory`: `SingleTurn` vs `Convo`) and a `feedback_fn` handler to monitor how a "participant's" beliefs, context, or fatigue might dynamically change from trial to trial across an experiment session.
4. Sandboxing vs. Logging
Inspect: Emphasizes heavily secure sandboxing (via Docker or Kubernetes) to safely execute untrusted code that an LLM generates during evaluation tasks.

PsychScanner: Doesn’t prioritize code execution infrastructure. Instead, it prioritizes the SessionTunnel which acts almost like a neurological/behavioral tracking log. It checkpoints cognitive processes, prompts, agent state, and creates .psyscan archives of simulated human-like responses.

Summary: If you wanted to test how well Claude or GPT-4 could write a Python script or navigate a website, you would use Inspect. If you wanted to simulate how a group of 50 anxious teenagers would rate their visualization capacity on a validated psychological questionnaire over 10 trials, you would use PsychScanner!