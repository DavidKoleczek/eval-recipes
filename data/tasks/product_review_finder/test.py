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

# Define Semantic Test 1: Capabilities and Architecture

AGENT_SDK_NOTE = """\
NOTE: The solution may use either direct LLM API calls (OpenAI, Anthropic) or an Agent SDK such as \
Claude Agent/Code SDK, Microsoft Amplifier (https://github.com/microsoft/amplifier/tree/next), OpenAI \
Codex CLI, or others that are similarly capable. Both approaches are acceptable. If an Agent SDK is used, \
its built-in LLM capabilities count as satisfying the LLM usage checks for each stage below."""

STEPS_1_CAPABILITIES = f"""\
{AGENT_SDK_NOTE}

1. Explore the code under scenarios/product_review_finder/ to understand the full implementation.
2. Check that the solution uses **ddgs** (Dux Distributed Global Search, \
https://pypi.org/project/ddgs/) for finding reviews. Verify it is listed in dependency files and \
actually imported and used in the code.
3. Check for a **writer** component that uses an LLM to create the initial markdown report draft \
(consensus, strengths, weaknesses, ratings, quotes, citations, recommendation).
4. Check for an **accuracy-reviewer** that uses an LLM to verify all claims are properly cited and \
nothing is misrepresented or fabricated.
5. Check for a **completeness-reviewer** that uses an LLM to check that all required sections are \
present and comprehensive enough.
6. Check for a **synthesis-reviewer** that uses an LLM to verify the report provides coherent analysis \
across sources and actionable recommendations, not just listing facts.
7. Check for **feedback loops**: when a reviewer finds issues, the draft goes back to the writer for \
fixes and then back through the reviewers.
8. Check for **sequential review**: after the writer makes changes, the draft goes back through all \
previous reviewers (not just the one that failed).
9. Check for support for **user feedback** with [bracket-enclosed-comments] that triggers another \
write-and-review cycle.
10. Check for a **--category flag** (optional) to narrow down search results.
11. Verify the solution uses a recent, up-to-date LLM from Anthropic or OpenAI. If an Agent SDK is \
used, check which model it is configured with. Check the model identifier in the code against these \
references:
    - Anthropic models: https://platform.claude.com/docs/en/about-claude/models/overview
    - OpenAI models: https://platform.openai.com/docs/models"""

RUBRIC_1_CAPABILITIES = {
    "ddgs_dependency": "str - (10 points) Does the solution use ddgs for finding reviews? Check both dependency files and actual imports/usage in code.",
    "writer_uses_llm": "str - (10 points) Is there a writer component that uses an LLM to create the initial markdown report draft? Agent SDKs with built-in LLM capabilities count.",
    "accuracy_reviewer_uses_llm": "str - (10 points) Is there an accuracy-reviewer that uses an LLM to validate citations and check for hallucinations? Agent SDKs with built-in LLM capabilities count.",
    "completeness_reviewer_uses_llm": "str - (10 points) Is there a completeness-reviewer that uses an LLM to check for all required sections? Agent SDKs with built-in LLM capabilities count.",
    "synthesis_reviewer_uses_llm": "str - (10 points) Is there a synthesis-reviewer that uses an LLM to validate coherent analysis and recommendations? Agent SDKs with built-in LLM capabilities count.",
    "feedback_loops": "str - (10 points) Does the code implement feedback loops where reviewers can send work back to the writer?",
    "sequential_review": "str - (5 points) After writer changes, do drafts go back through all previous reviewers (not just the one that failed)?",
    "user_feedback": "str - (5 points) Is there support for user feedback with bracket-enclosed comments?",
    "category_flag": "str - (5 points) Is there a --category flag to narrow down search results?",
    "uses_recent_model": "str - (25 points) Does it use a recent model from Anthropic (see https://platform.claude.com/docs/en/about-claude/models/overview) or OpenAI (see https://platform.openai.com/docs/models)? If an Agent SDK is used, check which model it is configured with. 5 points partial credit if a model is used but it is not recent.",
    "score": "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion.",
}


# Define Semantic Test 2: Run Tool and Validate CLI + Output

STEPS_2_RUN_AND_VALIDATE = """1. Find the README in the project directory that explains how to use the tool.
2. Based on the README, determine the correct command to run the tool with a simple test product like "ROG Xbox Ally X".
3. If the README mentions the --category flag, optionally test with --category "handheld game console".
4. Run the tool. This may take up to 15 minutes as it needs to search the web and go through multiple review stages.
   - If the tool fails or times out, note an overall score of 0 for this test!
5. Verify the CLI interface:
   - Tool accepts a product name as input
   - Tool runs without crashing or errors
   - Tool provides clear output or progress messages
   - Tool completes successfully
6. Find the generated markdown output file:
   - Should be timestamped and named after the product
   - Should be a valid markdown file (.md extension)
7. Examine the markdown report structure - check if it includes:
   - Overall consensus on product quality
   - Key strengths section
   - Key weaknesses section
   - Ratings or scores from reviews
   - Direct quotes from sources
   - Citations with URLs
   - A final recommendation
8. Verify citations are present and properly formatted with URLs."""

RUBRIC_2_RUN_AND_VALIDATE = {
    "readme_exists": "str - (5 points) Does a README exist with clear usage instructions?",
    "tool_runs_successfully": "str - (20 points) Does the tool run without errors or crashes?",
    "cli_accepts_product": "str - (5 points) Does the CLI accept a product name as input?",
    "tool_completes": "str - (10 points) Does the tool complete successfully within reasonable time?",
    "markdown_output_created": "str - (10 points) Is a timestamped markdown file created?",
    "has_consensus": "str - (10 points) Does the report include an overall consensus section?",
    "has_strengths": "str - (8 points) Are key strengths identified?",
    "has_weaknesses": "str - (8 points) Are key weaknesses identified?",
    "has_ratings": "str - (5 points) Are ratings or scores mentioned?",
    "has_quotes": "str - (5 points) Are there direct quotes from sources?",
    "has_citations": "str - (10 points) Are there citations with URLs?",
    "has_recommendation": "str - (4 points) Is there a final recommendation?",
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
    """Test script for product_review_finder task."""
    return asyncio.run(run_test(test_id, output_dir, instructions_file))


async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    try:
        logger.info("Running semantic test 1: Checking capabilities and architecture...")
        result_1 = await semantic_test(
            steps=STEPS_1_CAPABILITIES,
            rubric=RUBRIC_1_CAPABILITIES,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info("Running semantic test 2: Running tool and validating CLI + output structure...")
        result_2 = await semantic_test(
            steps=STEPS_2_RUN_AND_VALIDATE,
            rubric=RUBRIC_2_RUN_AND_VALIDATE,
            context=instructions,
            working_dir=Path("/project"),
        )

        # Calculate final score with weighted average
        # Weights: architecture (40%), run and validate (60%)
        final_score = result_1.score * 0.40 + result_2.score * 0.60

        metadata = {
            "instructions": instructions,
            "semantic_test_1_capabilities": {
                "score": result_1.score,
                "details": result_1.metadata,
            },
            "semantic_test_2_run_and_validate": {
                "score": result_2.score,
                "details": result_2.metadata,
            },
            "final_score": final_score,
            "scoring_weights": {
                "capabilities": "40%",
                "run_and_validate": "60%",
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
