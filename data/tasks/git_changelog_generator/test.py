# Copyright (c) Microsoft. All rights reserved.

import asyncio
from pathlib import Path
import sys

import click
from loguru import logger

from eval_recipes.benchmarking.evaluation.semantic_test import semantic_test
from eval_recipes.benchmarking.evaluation.test_utils import (
    get_instructions_from_file_or_default,
    get_test_id_from_env_or_default,
    write_test_result,
)

AGENT_SDK_NOTE = """\
NOTE: The solution may use either direct LLM API calls (OpenAI, Anthropic) or an Agent SDK such as \
Claude Agent/Code SDK, Microsoft Amplifier (https://github.com/microsoft/amplifier/tree/next), OpenAI \
Codex CLI, or others that are similarly capable. Both approaches are acceptable. If an Agent SDK is used, \
its built-in LLM capabilities count as satisfying the LLM usage checks below."""

# Define Semantic Test 1: Code Review and Architecture

STEPS_1_CODE_REVIEW = f"""\
{AGENT_SDK_NOTE}

1. Explore the code in the project directory to understand the full implementation.
2. Check that the solution uses **git** operations (via subprocess, gitpython, or similar) to retrieve \
commits between two references. Verify it actually runs git commands uses a git library to get the commit log, 
or relies on the agent to make the git call (any of these are acceptable).
3. Check that the solution uses an **LLM to categorize and summarize** the commits into meaningful \
groups (not just dumping raw commit messages). This could be via direct API calls or an Agent SDK.
4. Check for a **CLI interface** that accepts:
   - A repository URL or local path
   - Two commit references (tags, SHAs, or branches)
   - An option to configure the output file path
5. Check that the output is a **markdown file** with:
   - Categorized sections (e.g., Features, Bug Fixes, etc.)
   - Summarized descriptions within each category
6. Check that a **README.md** exists with usage instructions and examples.
7. Check for reasonable **error handling** (invalid refs, empty ranges, network errors).
8. Verify the solution uses a recent, up-to-date LLM from Anthropic or OpenAI. If an Agent SDK is used, \
check which model it is configured with. Check the model identifier in the code against these references:
    - Anthropic models: https://platform.claude.com/docs/en/about-claude/models/overview
    - OpenAI models: https://platform.openai.com/docs/models"""

RUBRIC_1_CODE_REVIEW = {
    "git_operations": "str - (15 points) Does the solution use git operations (subprocess, gitpython, or similar) to retrieve commits between two references? Check for actual git commands or library usage in the code.",
    "llm_categorization": "str - (15 points) Does the solution use an LLM to categorize and summarize commits into meaningful groups? It should not just dump raw commit messages. Agent SDKs with built-in LLM capabilities count.",
    "cli_interface": "str - (10 points) Does the CLI accept a repository URL or local path, two commit references, and a configurable output path?",
    "markdown_output": "str - (10 points) Does the code produce a markdown file with categorized sections and summarized descriptions?",
    "readme_exists": "str - (10 points) Does a README.md exist with usage instructions and examples?",
    "error_handling": "str - (5 points) Does the code handle edge cases like invalid refs, empty ranges, or network errors?",
    "clean_architecture": "str - (10 points) Is the code well-structured with separation between git operations, LLM processing, and output formatting?",
    "uses_recent_model": "str - (25 points) Does it use a recent model from Anthropic (see https://platform.claude.com/docs/en/about-claude/models/overview) or OpenAI (see https://platform.openai.com/docs/models)? If an Agent SDK is used, check which model it is configured with. 5 points partial credit if a model is used but it is not recent.",
    "score": "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion.",
}

# Define Semantic Test 2: Functional Test Against microsoft/playwright

STEPS_2_FUNCTIONAL = """\
1. Find the README.md file that explains how to use the tool and what commands to run.
2. Based on the README, install any required dependencies if needed.
3. Run the tool against the microsoft/playwright repository with the range v1.45.0 to v1.47.0. \
The command should clone or reference the repo and generate a changelog between these two tags.
   - If the tool fails or errors out, note an overall score of 0 for this test.
   - This may take up to 20 minutes as it needs to clone the repo and process commits through an LLM.
4. After the tool completes, locate the generated markdown changelog file.
5. Read the official Playwright release notes that serve as ground truth for what actually changed:
   - /project/test_time_data/playwright_v1.46.0_release_notes.md (changes from v1.45.0 to v1.46.0)
   - /project/test_time_data/playwright_v1.47.0_release_notes.md (changes from v1.46.0 to v1.47.0)
6. Compare the generated changelog against the official release notes. Evaluate how well the \
generated changelog captures the major features, changes, and fixes documented in the official \
notes. The changelog does not need to match the official notes word-for-word, but the key themes \
and changes should be represented.
7. Check for content that is fabricated or hallucinated: does the generated changelog contain \
significant features or changes that are NOT in the official release notes or actual commits?"""

RUBRIC_2_FUNCTIONAL = {
    "tool_runs_successfully": "str - (20 points) Does the tool run without errors against microsoft/playwright v1.45.0..v1.47.0 and produce a markdown file?",
    "coverage_of_official_changes": "str - (50 points) How well does the generated changelog capture the major features and changes from the official release notes? Full points if most key items are represented. Partial credit proportionally.",
    "no_hallucinations": "str - (30 points) Is the changelog free of significant fabricated or hallucinated features that do not appear in the official release notes or actual commits?",
    "score": "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion.",
}


@click.command()
@click.option(
    "--test-id",
    default=lambda: get_test_id_from_env_or_default("dev"),
    help="Test ID for result file naming (defaults to EVAL_RECIPES_TEST_ID env var)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=lambda: Path(__file__).parents[0],
    help="Directory to write result file",
)
@click.option(
    "--instructions-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to instructions file (defaults to ./instructions.txt in working directory)",
)
def main(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    """Test script for git_changelog_generator task."""
    return asyncio.run(run_test(test_id, output_dir, instructions_file))


async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    try:
        logger.info("Running semantic test 1: Checking code architecture and implementation...")
        result_1 = await semantic_test(
            steps=STEPS_1_CODE_REVIEW,
            rubric=RUBRIC_1_CODE_REVIEW,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info("Running semantic test 2: Functional test against microsoft/playwright...")
        result_2 = await semantic_test(
            steps=STEPS_2_FUNCTIONAL,
            rubric=RUBRIC_2_FUNCTIONAL,
            context=instructions,
            working_dir=Path("/project"),
        )

        final_score = result_1.score * 0.40 + result_2.score * 0.60

        metadata = {
            "instructions": instructions,
            "semantic_test_1_code_review": {
                "score": result_1.score,
                "details": result_1.metadata,
            },
            "semantic_test_2_functional": {
                "score": result_2.score,
                "details": result_2.metadata,
            },
            "final_score": final_score,
            "scoring_weights": {
                "code_review": "40%",
                "functional": "60%",
            },
        }

        write_test_result(output_dir, test_id, final_score, metadata)
        logger.info(f"Test completed with final score: {final_score:.1f}/100")
        return 0

    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        metadata = {
            "instructions": instructions,
            "error": str(e),
        }
        write_test_result(output_dir, test_id, 0, metadata)
        return 0


if __name__ == "__main__":
    sys.exit(main())
