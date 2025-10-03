# Benchmarking

This module provides a benchmarking harness for evaluating AI agents on a variety of tasks.


## Creating a New Agent

Agents are defined in the `data/agents/` directory. Each agent is a subdirectory containing the files needed to install and run the agent.
Included agents are located in [data/agents/](../../data/agents/).

### Agent Directory Structure

```
data/agents/your_agent_name/
agent.yaml            # Agent configuration
install.dockerfile    # Docker commands to install the agent
command_template.txt  # Liquid template for running the agent
```

### File Descriptions

#### `agent.yaml`
YAML configuration file for the agent.

Optional fields:
- `required_env_vars`: List of environment variables required by the agent (e.g., `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`)

#### `install.dockerfile`
Contains Docker `RUN` commands to install the agent and its dependencies.

#### `command_template.txt`
Liquid template for the command that will execute the agent. Use `{{task_instructions}}` as a placeholder for task instructions.

## Creating a New Task

Tasks are defined in the `data/tasks/` directory. Each task is a subdirectory containing the files needed to define the task and test the agent's solution.
Included tasks are available at [data/tasks/](../../data/tasks/).

### Task Directory Structure

```
data/tasks/your_task_name/
task.yaml            # Task configuration (required)
instructions.txt     # Instructions given to the agent (required)
test.py              # Python script to test the agent's solution (required)
setup.dockerfile     # (Optional) Docker commands to set up the task environment
test_commands.sh     # (Optional) Bash script to run before test.py
```

### File Descriptions

#### `task.yaml` (Required)
YAML configuration file for the task.

Required fields:
- `task_info`: Object containing:
  - `difficulty`: One of `easy`, `medium`, or `hard`
  - `non_deterministic_evals`: Boolean indicating if test evaluations are non-deterministic

Optional fields:
- `required_env_vars`: List of environment variables required for the task (e.g., API keys for evaluation)

#### `setup.dockerfile` (Optional)
Contains Docker `RUN` commands to install any dependencies needed for the task or tests.

#### `test_commands.sh` (Optional)
A bash script that runs before `test.py`. This is useful for installing test dependencies using `uv add` or setting up test data or configuration.

The script is executed from `/project` in the container. If it exists, the harness will:
1. Copy it into the container
2. Execute it with `bash test_commands.sh`
3. Save output to `test_commands_output.log`
4. Then proceed to run `test.py`

#### `instructions.txt` (Required)
Plain text instructions that will be passed to the agent. This describes what the agent should build or solve.

#### `test.py` (Required)
A Python test script that validates the agent's solution and outputs a score.

Critical Requirements:
1. Must read the `EVAL_RECIPES_TEST_ID` environment variable
2. Must write results to `.eval_recipes_test_results_{EVAL_RECIPES_TEST_ID}.json`. This JSON must contain `score` (0-100) and `metadata` (dict)
3. Must include a `main()` function that returns an exit code
