# Results: Reward-Learning Behavior Across Agent Architectures

We ran a 3-armed bandit task (arms A/B/C, hidden reward means 2.0/6.0/4.0
with unit Gaussian noise, `Q[a] += 0.3 * (reward - Q[a])` value updates fed
back to the model each round) through three `psychscanner.agents`
implementations plugged into the same `ExpCard`/`ScannerModel` pipeline via
`ScannerModel.run(custom_agent=...)`: a bare single-call baseline
(`CustomAgent`), a generate/reflect self-critique loop
(`make_basic_reflection_agent`), and a plan/execute/validate loop
(`make_planner_executor_agent`, arxiv:2310.00194). All three used the same
underlying model, `llama-3.1-8b-instant` on Groq, temperature 0.7, real API
calls throughout (no mocking). Each agent played 6 rounds for 1 simulated
participant, for 18 total scored rounds.

**Reward accumulation.** The plain baseline accumulated the most total
reward (16.14 over 6 rounds, mean 2.69/round), followed by basic reflection
(14.14 total, mean 2.36/round) and planner/executor (12.14 total, mean
2.02/round). None of the three came close to the optimal arm's true mean of
6.0, and all three underperformed even the mid-value arm C (mean 4.0) --
every agent spent most of its rounds on arm A, the *worst* of the three
options (true mean 2.0). Arm-selection counts: custom_direct chose A 3
times and C 3 times (0% optimal-arm picks); basic_reflection chose A 5
times and B once (16.7% optimal); planner_executor chose A 5 times and C
once (0% optimal). In the late-game window (last 2 of 6 rounds),
basic_reflection was the only agent to touch the optimal arm at all
(50% of late rounds on B, late mean reward 3.46), while custom_direct and
planner_executor stayed on non-optimal arms throughout (late means 2.46 and
1.46 respectively, 0% optimal-arm picks for both).

**Behavioral difference, and a caveat that matters more than the reward
numbers.** Reading the raw transcripts, the instruction ("state your final
choice clearly as a single letter") was reliably followed only in round 1
of each run (e.g. `custom_direct`'s first reply is literally "I'll choose
A."). From round 2 onward, once the JSON-formatted feedback block was
injected into the prompt, `llama-3.1-8b-instant` almost always responded
with multi-paragraph meta-commentary describing the JSON structure itself
("It appears to be a JSON object representing a trial in an
experiment...") rather than making a clean choice -- this was true across
all three agent types, not just the single-call baseline. `BanditFeedback`'s
arm parser (`re.findall(r"\b([ABC])\b", text.upper())`, taking the last
letter mentioned) still extracts *a* letter from these rambling responses,
so trials still scored, but the scored "choice" is confounded with whatever
letter the model happened to mention last while narrating the JSON, not
necessarily a deliberate decision. This is consistent with the cautionary
note already in `examples/17_cognitive_rl_bandits_and_pd.ipynb` ("don't
expect textbook uniform exploration... a small local model often locks onto
[a pattern] rather than sampling every arm") -- here the confound is
compounded by the model not reliably producing parseable output at all with
this small/fast model, an unflattering but real finding about instruction-
following degrading once feedback is injected into the prompt, independent
of which agent architecture wraps the call.

**Takeaways.** (1) The demonstration confirms the mechanical claim: three
structurally different `psychscanner.agents` implementations (bare call,
self-reflection, plan/execute/validate) all plug into the identical
`ExpCard`/`ScannerModel`/`FeedbackBase` reward pipeline with zero changes to
the task or feedback code -- only the `custom_agent=` argument to
`scanner.run()` differs. (2) None of the three architectures produced
textbook explore-then-exploit behavior on this small a sample; all
under-selected the optimal arm, and basic_reflection's extra self-critique
turn was the only one to reach the optimal arm during the late-game window.
(3) The dominant behavioral effect observed was not architecture but
instruction-following collapse under the JSON feedback format on this
8B-parameter model -- a confound future runs on this task should address
(e.g. plain-text feedback instead of JSON, or a larger model) before reading
too much into the per-agent reward gap above.

## Caveats / scope

This is a small demonstration run (`N_ROUNDS=6`, `NSIM=1` per agent, 18
scored rounds total), not a statistically powered study -- the numbers above
describe what this one run did, not a general claim about these
architectures. `NSIM` was reduced from an initially planned 3 down to 1, and
rounds from 12 to 6, after the first attempt (12 rounds x 3 sims) hit Groq's
free-tier rate limit (6000 tokens/min) partway through the `basic_reflection`
agent and crashed with an unhandled `groq.RateLimitError`; the simulation
script (`simulation/run_reward_task.py`) now wraps the model in a
retry-with-backoff shim (`RetryModel`, 25s backoff on 429) so a transient
rate limit no longer kills the run, and the smaller config keeps total token
usage comfortably under quota. Re-running with more rounds/participants (and
a paid Groq tier, or a plain-text rather than JSON feedback format) would
give a sturdier read on the architecture comparison itself.
