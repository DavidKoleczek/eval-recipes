# Copyright (c) Microsoft. All rights reserved.

from datetime import UTC, datetime
import io
from io import BytesIO
import json
from pathlib import Path
import tarfile
from typing import Any
import uuid

import docker
from liquid import Template
from loguru import logger
import yaml

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
            runs_dir: Path to runs directory (default: data/benchmarking/runs/<timestamp>/)
            environment: Environment variables to pass to containers
        """
        repo_root = Path(__file__).parents[2]
        self.agents_dir = agents_dir or repo_root / "data" / "agents"
        self.tasks_dir = tasks_dir or repo_root / "data" / "tasks"
        if runs_dir is None:
            timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
            self.runs_dir = repo_root / "data" / "benchmarking" / "runs" / timestamp
        else:
            self.runs_dir = runs_dir
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.base_template = Path(__file__).parents[0] / "base.dockerfile"
        self.environment = environment or {}

    def _load_agents(self) -> list[AgentConfig]:
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

    def _run_tests(self, container: Any, task: TaskConfig, run_dir: Path) -> TestResult | None:
        """Run test script in container and return results."""
        try:
            # Generate unique test ID
            test_id = str(uuid.uuid4())
            logger.info(f"Running tests with ID: {test_id}")

            # Create tar archive with test script and optional test_commands.sh
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                # Add test.py
                test_script_content = task.test_script.read_bytes()
                tarinfo = tarfile.TarInfo(name="test.py")
                tarinfo.size = len(test_script_content)
                tar.addfile(tarinfo, io.BytesIO(test_script_content))

                # Add test_commands.sh if it exists
                if task.test_commands_script and task.test_commands_script.exists():
                    test_commands_content = task.test_commands_script.read_bytes()
                    tarinfo = tarfile.TarInfo(name="test_commands.sh")
                    tarinfo.size = len(test_commands_content)
                    tarinfo.mode = 0o755  # Make it executable
                    tar.addfile(tarinfo, io.BytesIO(test_commands_content))

            tar_stream.seek(0)

            # Copy files into container
            container.put_archive("/project", tar_stream)
            logger.info("Copied test script into container")

            # Run test_commands.sh if it exists
            if task.test_commands_script and task.test_commands_script.exists():
                logger.info("Running test_commands.sh...")
                commands_output_file = run_dir / "test_commands_output.log"
                exec_result = container.exec_run(
                    cmd=["bash", "/project/test_commands.sh"],
                    stream=True,
                    demux=True,
                    workdir="/project",
                )

                # Collect and save test_commands.sh output
                with commands_output_file.open("wb") as f:
                    for chunk in exec_result.output:
                        if chunk:
                            if isinstance(chunk, tuple):
                                stdout, stderr = chunk
                                if stdout:
                                    f.write(stdout)
                                if stderr:
                                    f.write(stderr)
                            else:
                                f.write(chunk)

                logger.info(f"test_commands.sh output saved to: {commands_output_file}")

            # Execute test using uv run with test ID environment variable
            logger.info("Running tests with uv...")
            exec_result = container.exec_run(
                cmd=["uv", "run", "--no-project", "/project/test.py"],
                stream=True,
                demux=True,
                environment={"EVAL_RECIPES_TEST_ID": test_id},
            )

            # Collect output
            test_output_lines = []
            test_output_file = run_dir / "test_output.log"
            with test_output_file.open("wb") as f:
                for chunk in exec_result.output:
                    if chunk:
                        if isinstance(chunk, tuple):
                            stdout, stderr = chunk
                            if stdout:
                                f.write(stdout)
                                test_output_lines.append(stdout.decode("utf-8", errors="ignore"))
                            if stderr:
                                f.write(stderr)
                                test_output_lines.append(stderr.decode("utf-8", errors="ignore"))
                        else:
                            f.write(chunk)
                            test_output_lines.append(chunk.decode("utf-8", errors="ignore"))

            logger.info(f"Test output saved to: {test_output_file}")
            full_output = "".join(test_output_lines)

            # Read result file from container
            result_file_path = f"/project/.eval_recipes_test_results_{test_id}.json"
            result = container.exec_run(["cat", result_file_path])

            if result.exit_code == 0:
                logger.info("Successfully read results file from container")
                result_data = json.loads(result.output)
                test_result = TestResult(
                    score=result_data["score"],
                    metadata=result_data.get("metadata", {}),
                    test_output=full_output,
                )

                # Save test results to JSON file
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

        client = docker.from_env()
        try:
            for agent in agents:
                for task in tasks:
                    logger.info(f"Running agent '{agent.name}' on task '{task.name}'")

                    # Validate required environment variables
                    valid, missing_vars = self._validate_required_env_vars(agent, task)
                    if not valid:
                        logger.error(
                            f"Missing required environment variables for agent '{agent.name}' "
                            f"and task '{task.name}': {missing_vars}"
                        )
                        continue

                    dockerfile_content = self._build_dockerfile(agent, task)

                    image_tag = f"benchmark-{agent.name}-{task.name}".lower()
                    try:
                        _, _ = client.images.build(
                            fileobj=BytesIO(dockerfile_content.encode()),
                            tag=image_tag,
                            rm=True,
                        )

                        # Get environment variables to pass to container
                        container_env = self._get_container_env_vars(agent, task)

                        container = client.containers.run(
                            image=image_tag,
                            detach=True,
                            environment=container_env,
                            tty=True,
                            stdin_open=True,
                        )
                        if container:
                            container_id = getattr(container, "id", None)
                            if container_id:
                                logger.info(f"Container {container_id[:12]} started")

                            # Create run directory
                            run_dir = self.runs_dir / f"{agent.name}_{task.name}"
                            run_dir.mkdir(parents=True, exist_ok=True)
                            output_file = run_dir / "output.log"

                            # Render command from template
                            command_template = Template(agent.command_template)
                            command = command_template.render(task_instructions=task.instructions)

                            logger.info(f"Executing command: {command}")
                            logger.info(f"Streaming output to: {output_file}")

                            # Execute command and stream output
                            exec_result = container.exec_run(
                                cmd=["bash", "-c", command],
                                stream=True,
                                demux=True,
                            )

                            # Stream output to file
                            with output_file.open("wb") as f:
                                for chunk in exec_result.output:
                                    if chunk:
                                        if isinstance(chunk, tuple):
                                            # demux=True returns (stdout, stderr) tuples
                                            stdout, stderr = chunk
                                            if stdout:
                                                f.write(stdout)
                                            if stderr:
                                                f.write(stderr)
                                        else:
                                            f.write(chunk)

                            logger.info("Command execution completed")

                            # Get final logs
                            logs = container.logs()
                            if logs:
                                logs_file = run_dir / "container.log"
                                logs_file.write_bytes(logs)
                                logger.info(f"Container logs saved to: {logs_file}")

                            # Run tests
                            test_result = self._run_tests(container, task, run_dir)
                            if test_result:
                                logger.info(f"Tests completed - Score: {test_result.score}")
                            else:
                                logger.warning("Tests failed or could not be parsed")

                            # Clean up container
                            try:
                                container.remove(force=True)
                                if container_id:
                                    logger.info(f"Container {container_id[:12]} removed")
                            except Exception as e:
                                logger.warning(f"Failed to remove container: {e}")
                    finally:
                        try:
                            client.images.remove(image_tag, force=True)
                        except Exception as e:
                            logger.warning(f"Failed to remove image {image_tag}: {e}")
        finally:
            client.close()


if __name__ == "__main__":
    import asyncio
    import os

    from dotenv import load_dotenv

    load_dotenv()

    harness = Harness(
        environment={
            "ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"],
            "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
        }
    )
    asyncio.run(harness.run())
