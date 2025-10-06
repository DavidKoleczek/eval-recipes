# Copyright (c) Microsoft. All rights reserved.

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
import uuid

from liquid import Template
from loguru import logger
import yaml

from eval_recipes.benchmarking.docker_manager import DockerManager
from eval_recipes.benchmarking.schemas import AgentConfig, TaskConfig, TaskInfo, TestResult


class Harness:
    def __init__(
        self,
        agents_dir: Path | None = None,
        tasks_dir: Path | None = None,
        runs_dir: Path | None = None,
        environment: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize the benchmark harness.

        Args:
            agents_dir: Path to agents directory (default: data/agents/)
            tasks_dir: Path to tasks directory (default: data/tasks/)
            runs_dir: Path to base runs directory (default: data/benchmarking/runs/).
                     A timestamped subdirectory will be created under this path.
            environment: Environment variables to pass to containers
        """
        repo_root = Path(__file__).parents[2]
        self.agents_dir = agents_dir or repo_root / "data" / "agents"
        self.tasks_dir = tasks_dir or repo_root / "data" / "tasks"

        # Always create a timestamped directory under runs_dir
        base_runs_dir = runs_dir or repo_root / "data" / "benchmarking" / "runs"
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        self.runs_dir = base_runs_dir / timestamp
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        self.base_template = Path(__file__).parents[0] / "base.dockerfile"
        self.environment = environment or {}

    def _load_agents(self) -> list[AgentConfig]:
        """
        Loads agent configurations from the agents directory.
        """
        agents = []
        if not self.agents_dir.exists():
            logger.warning(f"Agents directory {self.agents_dir} does not exist.")
            return agents

        for agent_dir in self.agents_dir.iterdir():
            if not agent_dir.is_dir():
                continue

            install_file = agent_dir / "install.dockerfile"
            command_template_file = agent_dir / "command_template.txt"
            agent_yaml_file = agent_dir / "agent.yaml"

            if not install_file.exists() or not command_template_file.exists() or not agent_yaml_file.exists():
                continue

            with agent_yaml_file.open() as f:
                agent_yaml = yaml.safe_load(f) or {}

            agents.append(
                AgentConfig(
                    name=agent_dir.name,
                    required_env_vars=agent_yaml.get("required_env_vars", []),
                    agent_installation=install_file.read_text(),
                    command_template=command_template_file.read_text(),
                )
            )

        return agents

    def _load_tasks(self) -> list[TaskConfig]:
        """
        Loads task configurations from the tasks directory.
        """
        tasks = []
        if not self.tasks_dir.exists():
            logger.warning(f"Tasks directory {self.tasks_dir} does not exist.")
            return tasks

        for task_dir in self.tasks_dir.iterdir():
            if not task_dir.is_dir():
                continue

            setup_file = task_dir / "setup.dockerfile"
            instructions_file = task_dir / "instructions.txt"
            test_script = task_dir / "test.py"
            test_commands_script = task_dir / "test_commands.sh"
            task_yaml_file = task_dir / "task.yaml"

            if not instructions_file.exists() or not test_script.exists() or not task_yaml_file.exists():
                continue

            with task_yaml_file.open() as f:
                task_yaml = yaml.safe_load(f) or {}

            task_info_data = task_yaml.get("task_info")
            if not task_info_data:
                logger.warning(f"Skipping task '{task_dir.name}', missing required 'task_info' field in task.yaml")
                continue

            task_info = TaskInfo(
                difficulty=task_info_data["difficulty"],
                non_deterministic_evals=task_info_data["non_deterministic_evals"],
            )

            tasks.append(
                TaskConfig(
                    name=task_dir.name,
                    required_env_vars=task_yaml.get("required_env_vars", []),
                    task_installation=setup_file.read_text() if setup_file.exists() else "",
                    instructions=instructions_file.read_text(),
                    test_script=test_script,
                    test_commands_script=test_commands_script if test_commands_script.exists() else None,
                    task_info=task_info,
                )
            )

        return tasks

    def _validate_required_env_vars(self, agent: AgentConfig, task: TaskConfig) -> tuple[bool, list[str]]:
        """
        Validate that all required environment variables are provided.

        Returns:
            Tuple of (success, missing_vars) where success is True if all required vars are present
        """
        required_vars = set(agent.required_env_vars + task.required_env_vars)
        missing_vars = [var for var in required_vars if var not in self.environment]
        return len(missing_vars) == 0, missing_vars

    def _get_container_env_vars(self, agent: AgentConfig, task: TaskConfig) -> dict[str, str]:
        """
        Get the environment variables to pass to the container.

        Returns only the environment variables that are required by the agent or task.
        """
        required_vars = set(agent.required_env_vars + task.required_env_vars)
        return {var: self.environment[var] for var in required_vars if var in self.environment}

    def _build_dockerfile(self, agent: AgentConfig, task: TaskConfig) -> str:
        """Build the complete Dockerfile from base template using liquid."""
        base_template = self.base_template.read_text()
        template = Template(base_template)
        return template.render(
            agent_installation=agent.agent_installation,
            task_installation=task.task_installation,
        )

    def _run_tests(
        self, container: Any, task: TaskConfig, run_dir: Path, docker_manager: DockerManager
    ) -> TestResult | None:
        """Run test script in container and return results."""
        try:
            # Generate unique test ID - this is to make sure we can identify result files uniquely in the container
            test_id = str(uuid.uuid4())
            logger.info(f"Running tests with ID: {test_id}")

            # Copy the test script and the installation script (if any)
            files = {"test.py": task.test_script.read_bytes()}
            executable_files = set()
            if task.test_commands_script and task.test_commands_script.exists():
                files["test_commands.sh"] = task.test_commands_script.read_bytes()
                executable_files.add("test_commands.sh")
            docker_manager.copy_files_to_container(
                container=container, files=files, dest_path="/project", executable_files=executable_files
            )

            # Run test install script
            if task.test_commands_script and task.test_commands_script.exists():
                _exec_result, _commands_output = docker_manager.exec_command(
                    container=container,
                    command=["bash", "/project/test_commands.sh"],
                    log_filename="test_install_output.log",
                    workdir="/project",
                )
                logger.info(f"test_commands.sh output saved to: {run_dir / 'test_install_output.log'}")

            # Execute test using uv run with test ID environment variable
            _exec_result, full_output = docker_manager.exec_command(
                container=container,
                command=["uv", "run", "--no-project", "/project/test.py"],
                log_filename="test_output.log",
                environment={"EVAL_RECIPES_TEST_ID": test_id},
            )
            logger.info(f"Test output saved to: {run_dir / 'test_output.log'}")

            # Read result file from container
            result_file_path = f"/project/.eval_recipes_test_results_{test_id}.json"
            result_output = docker_manager.read_file_from_container(container, result_file_path)
            if result_output:
                result_data = json.loads(result_output)
                test_result = TestResult(
                    score=result_data["score"],
                    metadata=result_data.get("metadata", {}),
                    test_output=full_output,
                )
                results_file = run_dir / "test_results.json"
                results_file.write_text(json.dumps(result_data, indent=2))
                logger.info(f"Test score: {test_result.score}, metadata: {test_result.metadata}")
                return test_result
            else:
                logger.warning(f"Could not read results file: {result_file_path}")
                # Fallback: return score 0 with error metadata
                result_data = {"score": 0, "metadata": {"error": "No results file found"}}
                test_result = TestResult(
                    score=0,
                    metadata=result_data["metadata"],
                    test_output=full_output,
                )
                return test_result
        except Exception as e:
            logger.error(f"Failed to run tests: {e}")
            return None

    async def run(self) -> None:
        agents = self._load_agents()
        tasks = self._load_tasks()

        for agent in agents:
            for task in tasks:
                logger.info(f"Running agent '{agent.name}' on task '{task.name}'")

                valid, missing_vars = self._validate_required_env_vars(agent, task)
                if not valid:
                    logger.error(
                        f"Missing required environment variables for agent '{agent.name}' "
                        f"and task '{task.name}': {missing_vars}"
                    )
                    continue

                run_dir = self.runs_dir / f"{agent.name}_{task.name}"
                run_dir.mkdir(parents=True, exist_ok=True)

                container_env = self._get_container_env_vars(agent, task)
                dockerfile_content = self._build_dockerfile(agent, task)
                image_tag = f"benchmark-{agent.name}-{task.name}".lower()
                with DockerManager(
                    log_dir=run_dir, dockerfile=dockerfile_content, image_tag=image_tag, container_env=container_env
                ) as docker_manager:
                    assert docker_manager.container is not None
                    logger.info(f"Built image: {docker_manager.actual_image_tag}")
                    logger.info(f"Container {docker_manager.container_id} started")

                    # Create command to run agent
                    command_template = Template(agent.command_template)
                    command = command_template.render(task_instructions=task.instructions)
                    logger.info(f"Executing command: {command}")

                    _exec_result, _exec_logs = docker_manager.exec_command(
                        container=docker_manager.container,
                        command=["bash", "-c", command],
                        log_filename="agent_output.log",
                    )
                    logger.info(f"Command execution completed. Output saved to: {run_dir / 'agent_output.log'}")

                    self._run_tests(docker_manager.container, task, run_dir, docker_manager)
