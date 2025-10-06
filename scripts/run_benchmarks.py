# Copyright (c) Microsoft. All rights reserved

import asyncio
import os
from pathlib import Path

import click
from dotenv import load_dotenv

from eval_recipes.benchmarking.harness import Harness

load_dotenv()


@click.command()
@click.option(
    "--agents-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=lambda: Path(__file__).parents[1] / "data" / "agents",
    help="Directory containing agent configurations",
)
@click.option(
    "--tasks-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=lambda: Path(__file__).parents[1] / "data" / "tasks",
    help="Directory containing task definitions",
)
@click.option(
    "--runs-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=lambda: Path(__file__).parents[1] / "data" / "benchmarking" / "runs",
    help="Directory to store benchmark run results",
)
def main(agents_dir: Path, tasks_dir: Path, runs_dir: Path) -> None:
    """Run benchmarks for LLM agents."""
    harness = Harness(
        agents_dir=agents_dir,
        tasks_dir=tasks_dir,
        runs_dir=runs_dir,
        environment={
            "ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"],
            "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
        },
    )
    asyncio.run(harness.run())


if __name__ == "__main__":
    main()
