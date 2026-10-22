# paretochat

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**paretochat** is a multi-objective decision support library with an interactive LLM-powered assistant for exploring trade-offs in optimization problems.

It helps users navigate Pareto-optimal solution sets by expressing preferences in natural language. The assistant iteratively re-optimizes and presents solutions until the user is satisfied. This codebase implements the algorithm developed in the research paper:

```
@article{ceyhan2026paretochat,
  title={ParetoChat: An LLM-Based Interface for Interactive Multi-Objective Optimization},
  author={Ceyhan, Gokhan},
  journal={arXiv preprint arXiv:2610.23062},
  year={2026}
}
```


## Features

- Interactive trade-off exploration via a conversational agent (LangGraph-based)
- Pluggable LLM backends: Gemini (Vertex AI or direct API), Claude, or Ollama
- Example optimization problems: discrete choice, multi-objective knapsack, courier order assignment
- Simulation framework for benchmarking against DESDEO (RPM, NIMBUS) methods

## Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) (recommended for environment management)
- An LLM provider (see [LLM Configuration](#llm-configuration) below)

## Installation

From source:

```bash
git clone https://github.com/your-org/paretochat.git
cd paretochat
task setup
```

This creates a virtual environment, installs all dependencies, and installs the pre-commit hook that runs linting checks before each commit.

## LLM Configuration

Set the `LLM_PROVIDER` environment variable to select your backend:

| `LLM_PROVIDER` value | Provider | Required env vars | Model env var (optional) |
|---|---|---|---|
| `gemini-vertex` (default) | Gemini via Vertex AI | `GOOGLE_CLOUD_PROJECT` | `GEMINI_VERTEX_MODEL` (default: `gemini-2.5-pro`) |
| `gemini-api` | Gemini via Google AI API | `GOOGLE_API_KEY` | `GEMINI_API_MODEL` (default: `gemini-2.0-flash`) |
| `claude-cli` | Claude via Claude CLI | Claude CLI installed and authenticated | `CLAUDE_MODEL` (default: `claude-sonnet-4-5`) |
| `ollama` | Local Ollama server | Ollama running locally | `OLLAMA_MODEL` (default: `llama3.1:8b`), `OLLAMA_BASE_URL` (default: `http://localhost:11434`) |

### Vertex AI setup

`gemini-vertex` uses [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials). For local development, authenticate with the gcloud CLI:

```bash
# Install gcloud CLI: https://cloud.google.com/sdk/docs/install
gcloud auth application-default login

export LLM_PROVIDER=gemini-vertex
export GOOGLE_CLOUD_PROJECT=your-gcp-project-id
export GOOGLE_CLOUD_REGION=us-west1  # optional, defaults to us-west1
export GEMINI_VERTEX_MODEL=gemini-2.5-flash  # optional, defaults to gemini-2.5-pro
```

### Ollama setup

```bash
# Install Ollama: https://ollama.com/download
ollama pull llama3.1:8b  # or any tool-calling model

export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.1:8b        # optional, defaults to llama3.1:8b
export OLLAMA_BASE_URL=http://localhost:11434  # optional, for remote Ollama servers
```

Models with structured output and tool calling support: `qwen2.5:7b`, `qwen2.5:14b`, `llama3.1:8b`, `llama3.3:70b`, `mistral-nemo`.

> **Note:** Initial tests show that even the models listed above do not perform reliably enough for paretochat's structured output and tool calling requirements. The `ollama` provider is experimental — prefer `gemini-vertex`, `gemini-api`, or `claude-cli`.


## Quick Start

## Available Examples

All examples start an interactive CLI for exploring trade-offs.

**Approval of algorithm runs:** before the algorithm is run, ParetoChat has a human-in-the-loop approval step for the scalarization proposed by the LLM. By default, the examples auto-approve every run. Pass `--no-auto-approve` to be shown each proposed scalarization and confirm it before the algorithm runs (rejecting cancels the run):

```bash
uv run python paretochat/examples/biobj_discrete_choice/main.py --no-auto-approve
```

When using `ParetoChatAgent` directly, pass `approval_handler=` (e.g. `confirm_run` from `paretochat.api.cli.interface`); the default is to approve all runs.

### 1) Bi-objective Discrete Choice

Choosing among pre-computed alternatives with two competing objectives: minimizing `cost` and minimizing `delivery_time`.

```bash
uv run python paretochat/examples/biobj_discrete_choice/main.py
```

### 2) Three-objective Knapsack

Solving the multi-objective knapsack problem with OR-Tools, balancing `total_value`, `total_utility`, and `total_happiness`.

```bash
uv run python paretochat/examples/three_obj_knapsack/main.py
```

> **Note:** This example uses the HiGHS solver, which is not bundled with OR-Tools by default. Install it separately (e.g. `brew install highs` on macOS or `apt install highs` on Debian/Ubuntu) and ensure the `highs` binary is on your `PATH`.

### 3) Courier Order Assignment

Multi-objective courier assignment solved by an Assignment Solver, trading off pickup duration, delivery delay, and pickup delay.

```bash
# Use defaults (looks for the JAR and instance/ directory alongside the script)
uv run python paretochat/examples/courier_order_assignment/main.py

# Specify paths explicitly
uv run python paretochat/examples/courier_order_assignment/main.py \
  --instance-dir /path/to/instances \
  --jar-path /path/to/solver.jar \
  --output-csv results.csv
```

> **Note:** This example requires Java 17+ on your `PATH` and a solver JAR. By default it looks for `solver-analytics-1.0.0-SNAPSHOT-jar-with-dependencies.jar` next to the script; use `--jar-path` to point to a different location. Ask the author to share this jar file and example instances if you are a JET employee and want to run this example.

## Shared CLI

Run any example through the shared CLI:

```bash
uv run python paretochat/api/cli/main.py --help
uv run python paretochat/api/cli/main.py biobj-discrete-choice
uv run python paretochat/api/cli/main.py three-obj-knapsack --no-auto-approve
uv run python paretochat/api/cli/main.py courier-order-assignment \
  --instance-dir /path/to/instances \
  --jar-path /path/to/solver.jar
```

## Simulations with AI Decision Maker

Run batch simulations to benchmark paretochat against DESDEO methods:

```bash
uv run python paretochat/simulation/main.py run-batch \
  --problem-type knapsack \
  --problem-path paretochat/simulation/instances/mobkp_instances/random/2D \
  --max-instances 1
```

See the simulation module for full options (`--help`).

### Benchmark Instances

The multi-objective knapsack instances under [`paretochat/simulation/instances/mobkp_instances/`](paretochat/simulation/instances/mobkp_instances/) are taken from Gonçalo Lopes' [mobkp-instances](https://github.com/gaplopes/mobkp-instances) repository. They are © 2024 Gonçalo Lopes and distributed under the MIT License, not this project's Apache 2.0 license. The license text is in [`mobkp_instances/LICENSE`](paretochat/simulation/instances/mobkp_instances/LICENSE) and the citation is in [`mobkp_instances/citation.md`](paretochat/simulation/instances/mobkp_instances/citation.md). If you use these instances in your work, please cite the original source.

Likewise, the set-partitioning instances and their nondominated sets under [`paretochat/simulation/instances/spa_instances/`](paretochat/simulation/instances/spa_instances/) are taken from [vOptLib](https://github.com/vOptSolver/vOptLib). They are © 2017 Xavier Gandibleux and contributors and distributed under the MIT License, not this project's Apache 2.0 license. The license text is in [`spa_instances/LICENSE`](paretochat/simulation/instances/spa_instances/LICENSE) and the source and reference are in [`spa_instances/instance_info.md`](paretochat/simulation/instances/spa_instances/instance_info.md). If you use these instances in your work, please cite the original source.

### Simulation Runtime Dependency

The DESDEO simulation backends require external solver binaries (`cbc`, `ipopt`) on your `PATH`.
See [DESDEO third-party optimizers](https://desdeo.readthedocs.io/en/latest/howtoguides/installing/#third-party-optimizers).

The simulation runs use Gurobi as the optimization solver. Gurobi is a commercial solver and requires a valid license — see [Gurobi licensing](https://www.gurobi.com/solutions/licensing/) for options (free academic licenses are available).

Set these environment variables before running the simulations:

```bash
export GUROBI_HOME="/path/to/gurobi"
export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
```

> OR-Tools 9.14 is required for compatibility with Gurobi 13 (see [discussion](https://github.com/google/or-tools/discussions/5026)).

> OR-Tools 9.14 pins `protobuf<6.32`, which is affected by CVE-2026-0994 (DoS in `json_format.ParseDict()`, fixed in protobuf 6.33.5). Upgrading OR-Tools to 9.15 is not an option because it breaks Gurobi. Instead, `[tool.uv] override-dependencies` in `pyproject.toml` forces `protobuf>=6.33.5,<7`. Remove this override once an OR-Tools release with Gurobi 13 support (see [PR #5272](https://github.com/google/or-tools/pull/5272)) is available and can be upgraded to.
