# Project: Latent & Flow Actor Extensions for SAC

This subpackage implements two actor variants that extend the standard Gaussian actor used in Soft Actor-Critic (SAC) style agents. The goal of both designs is to transform and enrich the action-noise distribution in a non-linear, learned fashion to improve stability and exploration.

High-level motivation
---------------------
- Standard stochastic policies in actor-critic methods sample actions from simple parametric distributions (e.g. diagonal Gaussian). While effective, these distributions can limit the kinds of exploration the agent can perform.
- By learning a richer, invertible or decodable transformation of the base noise distribution, we get more expressive action distributions while retaining tractable density evaluation via change-of-variables (important for off-policy algorithms that need log probabilities).
- This repository contains two approaches: a LatentActor (decoded latent action) and a FlowActor (normalizing-flow based transformation).

Implemented actors
------------------
- `LatentActor` (in `project/models/actor.py`)
  - Samples a latent action from the base diagonal Gaussian produced by the underlying `Actor`.
  - Optionally decodes the latent action through a small MLP decoder (conditional on features if `conditional_decoder=True`) to produce the final environment action.
  - Useful when you want a low-dimensional latent policy but richer decoding to the action space, or when you want to constrain the latent space (e.g., with `tanh` to keep values within [-1,1]).
  - Provides `get_latent_action`, `get_decoded_action`, and `action_log_prob` that return both a decoded action and the latent log-probability.

- `FlowActor` (in `project/models/actor.py`)
  - Samples a latent action from the base diagonal Gaussian.
  - Transforms that latent sample through a conditional or unconditional normalizing flow (RealNVP implementation provided in `project/models/flow.py`) to obtain a flexible final action distribution.
  - When using flows, the log-probability of the final action is obtained via change-of-variables: log p(x) = log p(z) - log|det df/dz| (or equivalently with the inverse Jacobian sign convention used by the flow implementation).
  - Provides `get_flow_action` and a custom `action_log_prob` that adjusts the latent log-prob with the flow log-determinant.

Why these designs?
-------------------
- Latent decoding keeps the policy in a compact latent space while allowing a non-linear mapping to the action — this can regularize learning and reduce variance of the policy network outputs.
- Normalizing flows give an exact (tractable) density under flexible transformations. They can represent multimodal and skewed distributions that diagonal Gaussians cannot.
- Both methods try to improve exploration by changing how stochasticity is injected into actions, and they can improve numerical stability when the decoder/flow is trained alongside the actor.

Quick usage notes
-----------------
- The actors extend the base `stable_baselines3` `Actor` class. They can be plugged into existing SAC training loops by replacing the policy actor with `LatentActor` or `FlowActor` variants where the rest of the agent expects the same interface.
- Key constructor args in `LatentActor` / `FlowActor`:
  - `latent_dim`: dimensionality of the latent action space (smaller => stronger bottleneck).
  - `latent_arch` / `flow_arch`: MLP sizes for decoder or flow hidden layers.
  - `conditional_decoder` / `conditional_flow`: whether to condition the decoder/flow on observation features.
  - `constrain_latent_space`: if True, applies `tanh` to mean actions to keep them in [-1, 1].

Log probability & flows
-----------------------
- Careful: the sign used when applying the flow log-determinant depends on the convention used in the `flow` implementation:
  - If your flow returns `logdet = log|det dx/dz|` (forward Jacobian), then use `log p(x) = log p(z) - logdet`.
  - If it returns `logdet = log|det dz/dx|` (inverse Jacobian), add the logdet instead.
- See the `FlowActor.action_log_prob` implementation for how the repository currently applies this term. If you want me to check `project/models/flow.py` and confirm the flow convention, I can do that and update the comment in code.

Where to look in the code
-------------------------
- `project/models/actor.py` : `LatentActor`, `FlowActor` implementations.
- `project/models/flow.py` : Flow layers and RealNVP / ConditionalRealNVP implementations.
- `project/models/policy.py` : Higher-level policy wiring that consumes actor outputs.
- `scripts/train.py` : Example training entrypoint (if present) showing how the model is instantiated for experiments.

Experiments and results
-----------------------
- No experimental results are included yet. Recommended first experiments:
  1. Compare SAC baseline (diagonal Gaussian actor) vs LatentActor vs FlowActor on a simple continuous control task (e.g., Pendulum / LunarLander).
  2. Ablate latent dimensionality and conditional vs unconditional decoders/flows.
  3. Track training stability (variance over seeds) and sample efficiency (reward vs wall-clock).

Next steps / TODO
-----------------
- Add clear unit-tests for the flow log-determinant sign and the `action_log_prob` correctness. A small density check (sample z -> x and compare log-prob via change-of-variables) is a cheap sanity test.
- Add CLI example or tutorial notebook showing how to instantiate and train with each actor.
- Run experiments and populate a `results/` subfolder with plots and metrics.

Inverse Kinematics environment (IK-RL)
--------------------------------------
This project uses a dedicated environment implementation for the 2D inverse kinematics benchmark. The environment is provided as the `ik_rl` subpackage inside `project/environment/ik_rl` and has its own packaging and documentation in that folder.

Install the environment

There are two ways to make the IK environment available to experiments:

1) Install the local package (recommended for development):

  - From the repository root run (uses Poetry-managed Python environment):

```bash
poetry install --no-root
poetry run pip install -e project/environment/ik_rl
```

  This installs the `ik_rl` package in editable mode so you can modify the environment code and test changes immediately.

2) Use PYTHONPATH during development (quick, no install):

```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)/project/environment/ik_rl"
python -m project ...
```

Usage via the project entrypoint
--------------------------------
The `project` package provides a small CLI wrapper that delegates to the `Entrypoint` class in `project/entrypoint.py`. Two main commands are provided:

- `train-sac` — start a SAC training run using the configured actor/policy.
- `render-sac` — render a trained checkpoint.

Examples (from repository root):

```bash
# Train (uses Hydra config at configs/train_sac.yaml)
python -m project train-sac --help
python -m project train-sac

# Render a checkpoint
python -m project render-sac --checkpoint path/to/checkpoint --device cpu
```

Internals: the CLI uses `pyargwriter`'s hydra wrapper to pass the Hydra-config to `Entrypoint.train_sac`. The `Entrypoint` class in `project/entrypoint.py` calls `project.scripts.train.train_sac(config, force, device)` under the hood.

Notes and troubleshooting
-------------------------
- Make sure the `ik_rl` environment is importable (either installed or PYTHONPATH set). Missing environment imports will raise at runtime when constructing envs in `scripts/train.py`.
- Use the `--help` flags to reveal available config overrides supported by `pyargwriter` and the entrypoint.
- The project relies on Poetry and specific pinned dependencies (see `pyproject.toml`). If you prefer pip/venv, install the packages listed under `[project].dependencies` in `pyproject.toml` into your virtualenv.

Development tips
----------------
- To iterate quickly on environments, use the editable install (`pip install -e project/environment/ik_rl`) and run the entrypoint from the repo root.
- Unit tests for the environment are available in `project/environment/ik_rl/tests/` and can be run with `pytest` once dependencies are installed.

License & contact
-----------------
This project uses the [MIT license](LICENSE). For questions, open an issue or contact the maintainer.
