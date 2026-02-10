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

# Define Semantic Test 1: Capabilities and Dependencies

AGENT_SDK_NOTE = """\
NOTE: The solution may use either direct LLM API calls (OpenAI, Anthropic) or an Agent SDK such as \
Claude Agent/Code SDK, Microsoft Amplifier (https://github.com/microsoft/amplifier/tree/next), OpenAI \
Codex CLI, or others that are similarly capable. Both approaches are acceptable. If an Agent SDK is used, \
its built-in LLM capabilities count as satisfying the LLM usage checks for each stage below."""

STEPS_1_CAPABILITIES = f"""\
{AGENT_SDK_NOTE}

1. Explore the code in the project directory to understand the full implementation.
2. Look for where dependencies are defined (e.g., pyproject.toml, requirements.txt, package.json, etc.)
3. Check that the solution uses the **ddgs** (Dux Distributed Global Search, \
https://pypi.org/project/ddgs/) Python library for web search. Verify it is listed in dependency files \
and actually imported and used in the code.
4. Check that the solution uses an **LLM to synthesize** the search results into a digestible report \
(not just raw search output). This could be via direct API calls or an Agent SDK.
5. Check for a **supporting information** stage: after the initial ddgs search returns links, the tool \
should find additional supporting information (e.g., fetching article content, follow-up searches).
6. Check that the output is saved to a **markdown file** with proper formatting.
7. Check that the CLI accepts a **topic as input** (command-line argument or option).
8. Check that a **README.md** exists with usage instructions and examples.
9. Verify the solution uses a recent, up-to-date LLM from Anthropic or OpenAI. If an Agent SDK is used, \
check which model it is configured with. Check the model identifier in the code against these references:
    - Anthropic models: https://platform.claude.com/docs/en/about-claude/models/overview
    - OpenAI models: https://platform.openai.com/docs/models"""

RUBRIC_1_CAPABILITIES = {
    "ddgs_dependency": "str - (25 points) Does the solution use the ddgs Python library for searching news? Check both dependency files and actual imports/usage in code.",
    "llm_synthesis": "str - (15 points) Does the solution use an LLM to synthesize search results into a digestible report (not just raw output)? Agent SDKs with built-in LLM capabilities count.",
    "supporting_information": "str - (10 points) After initial search, does the tool find supporting information (fetching article content, follow-up searches)?",
    "markdown_output": "str - (5 points) Does the tool save output to a properly formatted markdown file?",
    "cli_accepts_topic": "str - (5 points) Does the CLI accept a topic as input?",
    "readme_exists": "str - (5 points) Does a README.md exist with usage instructions and examples?",
    "grounded_citations": "str - (10 points) Does the implementation ensure content is grounded with proper citations including URLs and publication dates?",
    "uses_recent_model": "str - (25 points) Does it use a recent model from Anthropic (see https://platform.claude.com/docs/en/about-claude/models/overview) or OpenAI (see https://platform.openai.com/docs/models)? If an Agent SDK is used, check which model it is configured with. 5 points partial credit if a model is used but it is not recent.",
    "score": "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion.",
}


# Define Semantic Test 2: Run Tool and Validate Output

STEPS_2_RUN_AND_VALIDATE = """\
1. Find the README.md file that explains how to use the tool and what commands to run.
2. Based on the README:
   - Install any required dependencies if not already installed.
   - Determine the correct command to run the tool with this topic: \
"artificial intelligence breakthroughs 2025"
3. Run the tool. This may take up to 15 minutes as it needs to search the web and process results.
   - If the tool fails or errors out, note down an overall score of 0!
4. After completion, find the output file(s):
   - Should be a markdown file (.md extension) containing the research results
5. Evaluate the output structure:
   - Does the markdown file contain a summary that synthesizes findings?
   - Are there citations/references to source articles?
   - Do citations include URLs to the source articles?
   - Do citations include publication dates?
6. Evaluate the output quality:
   - Are the found articles actually relevant to AI breakthroughs?
   - Are the articles recent (2024-2025)?
   - Does the summary synthesize information from multiple sources (not just list them)?
   - Are there at least 3-5 different source articles cited?
7. Evaluate content grounding:
   - Is the summary based on actual information from the cited sources?
   - Are there any unsupported claims or hallucinations?
   - Does it properly attribute information to specific sources?"""

RUBRIC_2_RUN_AND_VALIDATE = {
    "readme_exists": "str - (5 points) Does a README.md exist with clear usage and installation instructions?",
    "tool_runs_successfully": "str - (20 points) Does the tool run without errors when given the topic?",
    "markdown_output_created": "str - (10 points) Is a markdown file created with the results?",
    "has_summary": "str - (10 points) Does the markdown file contain a summary of findings?",
    "articles_are_relevant": "str - (10 points) Are the found articles actually relevant to the topic?",
    "articles_are_recent": "str - (10 points) Are the articles recent (2024-2025)?",
    "synthesizes_multiple_sources": "str - (10 points) Does it synthesize information from multiple sources?",
    "has_multiple_citations": "str - (5 points) Are there at least 3-5 different source articles cited?",
    "citations_have_urls": "str - (5 points) Do citations include URLs to the source articles?",
    "citations_have_dates": "str - (5 points) Do citations include publication dates?",
    "content_is_grounded": "str - (5 points) Is the content grounded in actual source material without hallucinations?",
    "proper_attribution": "str - (5 points) Is information properly attributed to specific sources?",
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
    """Test script for news_research_tool task."""
    return asyncio.run(run_test(test_id, output_dir, instructions_file))


async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    try:
        logger.info("Running semantic test 1: Checking capabilities and dependencies...")
        result_1 = await semantic_test(
            steps=STEPS_1_CAPABILITIES,
            rubric=RUBRIC_1_CAPABILITIES,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info("Running semantic test 2: Running tool and validating output...")
        result_2 = await semantic_test(
            steps=STEPS_2_RUN_AND_VALIDATE,
            rubric=RUBRIC_2_RUN_AND_VALIDATE,
            context=instructions,
            working_dir=Path("/project"),
        )

        # Calculate final score with weighted average
        # Weights: capabilities (30%), run and validate (70%)
        final_score = result_1.score * 0.30 + result_2.score * 0.70

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
                "capabilities": "30%",
                "run_and_validate": "70%",
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
