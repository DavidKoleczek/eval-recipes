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

1. Explore the code to understand the full implementation.
2. Check that it reads past posts from a directory to learn the user's writing style and tone.
3. Check that it accepts topic notes as input.
4. Check for a **topic/metaphor discovery** stage that uses an LLM to pick topics and find a metaphor or \
story that makes technical concepts accessible.
5. Check for a **drafting** stage that uses an LLM to write creative paragraphs and draft explanations \
for any jargon.
6. Check for an **image generation** stage that uses an image generation API (OpenAI DALL-E 3, \
gpt-image-1, or similar) to actually produce images, not just create prompts. Verify imports and actual \
API calls in the code (e.g., openai.images.generate or similar).
7. Check for a **research/references** stage that uses an LLM to find references and incorporate them \
into the post.
8. Check for a **style matching** stage that uses an LLM to clean up the draft to match the user's \
previous writing style from past posts.
9. Check for a **review/editing** stage that uses an LLM to check logic and reasoning and do a final review.
10. Check for a social media version output option (a shortened version of the post).
11. Check that the implementation has separate stages or components -- not one monolithic prompt doing \
everything. This could be multiple agents, separate prompts, or modular functions.
12. Verify the solution uses a recent, up-to-date LLM from Anthropic or OpenAI. If an Agent SDK is used, \
check which model it is configured with. Check the model identifier in the code against these references:
    - Anthropic models: https://platform.claude.com/docs/en/about-claude/models/overview
    - OpenAI models: https://platform.openai.com/docs/models"""

RUBRIC_1_CAPABILITIES = {
    "reads_past_posts": "str - (5 points) Does the code read past posts from a directory to learn style/tone?",
    "accepts_topic_notes": "str - (5 points) Does the code accept topic notes as input?",
    "topic_metaphor_discovery": "str - (10 points) Is there a stage that uses an LLM to pick topics and find metaphors/stories for technical concepts? Agent SDKs with built-in LLM capabilities count.",
    "drafting_stage": "str - (10 points) Is there a stage that uses an LLM to write creative paragraphs and draft jargon explanations? Agent SDKs with built-in LLM capabilities count.",
    "image_generation": "str - (15 points) Does the solution use an image generation API (DALL-E 3, gpt-image-1, or similar) to actually produce images? Verify actual API calls, not just prompt creation.",
    "research_references": "str - (10 points) Is there a stage that uses an LLM to find references and incorporate them? Agent SDKs with built-in LLM capabilities count.",
    "style_matching": "str - (5 points) Is there a stage that uses an LLM to match the user's past writing style? Agent SDKs with built-in LLM capabilities count.",
    "review_editing": "str - (5 points) Is there a stage that uses an LLM to check logic/reasoning and do a final review? Agent SDKs with built-in LLM capabilities count.",
    "social_media_option": "str - (5 points) Does the tool have an option to output a shortened social media version?",
    "separate_stages": "str - (5 points) Does the implementation have separate stages/components (not one monolithic prompt)? Could be agentic loops, separate prompts, or modular functions.",
    "uses_recent_model": "str - (25 points) Does it use a recent model from Anthropic (see https://platform.claude.com/docs/en/about-claude/models/overview) or OpenAI (see https://platform.openai.com/docs/models)? If an Agent SDK is used, check which model it is configured with. 5 points partial credit if a model is used but it is not recent.",
    "score": "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion.",
}


# Define Semantic Test 2: Run Tool, Validate Outputs, and Assess Quality

STEPS_2_RUN_AND_VALIDATE = """1. Find the README or documentation that explains how to use the tool.
2. Locate the test data in the project:
   - past_posts/ directory with 3 sample posts
   - topic_notes.txt with notes about databases and controlled burns
3. Based on the README, determine the correct command to run the tool with these inputs.
4. Run the tool with the test data. This may take up to 15 minutes as it involves AI generation.
   - If the tool fails to complete after 15 minutes or errors out, note an overall score of 0 for this test!
5. After the tool completes, verify the CLI interface:
   - Does it accept past posts directory as input?
   - Does it accept topic notes as input?
   - Does it run without errors?
   - Does it have an option to output a social media version?
6. Check the outputs:
   - Is a post generated?
   - Are images or image prompts generated? Are the images embedded as Markdown links in the post?
   - Is there a social media version?
7. Analyze the writing style by reading all 3 sample posts carefully:
   - **MOST DISTINCTIVE CHARACTERISTIC**: Creative metaphors from completely unexpected domains
     * Post 1: Cicadas (biology) → Caching strategies (engineering)
     * Post 2: Medieval walled cities (history) → Microservices architecture (engineering)
     * Post 3: Jazz improvisation (music) → Debugging techniques (engineering)
   - Other style elements:
     * Short paragraphs with clear structure
8. Read the generated post and evaluate quality:
   - **CRITICAL**: Does it use a creative metaphor from an unexpected domain (not software/tech)?
     * The metaphor should come from biology, history, music, sports, nature, etc.
     * It should be used throughout the post to explain the technical concept
     * This is the signature characteristic - without this, it's not matching the style
   - Does it use short paragraphs with clear structure?"""

# SCORING GUIDANCE: The most distinctive characteristic of this writing style is the use of creative metaphors
# from completely unexpected domains (biology, history, music, etc.) to explain technical concepts.
# A post without this metaphor characteristic should receive low scores even if other elements are present.

RUBRIC_2_RUN_AND_VALIDATE = {
    "readme_exists": "str - (5 points) Does clear documentation exist explaining how to run the tool?",
    "accepts_past_posts": "str - (5 points) Does the CLI accept past posts as input?",
    "accepts_topic_notes": "str - (5 points) Does the CLI accept topic notes as input?",
    "tool_runs_successfully": "str - (20 points) Does the tool run without errors?",
    "generates_post": "str - (25 points) Does it generate a post?",
    "has_social_media_option": "str - (10 points) Does the tool have an option to output a social media version?",
    "uses_unexpected_metaphor": "str - (25 points) CRITICAL - Does the post use a creative metaphor from an unexpected domain (biology, history, music, sports, nature, etc.) to explain a technical concept? This is the signature characteristic of the style.",
    "short_paragraphs_structure": "str - (5 points) Does it use short paragraphs with clear structure?",
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
    """Test script for linkedin_drafting task."""
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

        logger.info("Running semantic test 2: Running tool, validating outputs, and assessing quality...")
        result_2 = await semantic_test(
            steps=STEPS_2_RUN_AND_VALIDATE,
            rubric=RUBRIC_2_RUN_AND_VALIDATE,
            context=instructions,
            working_dir=Path("/project"),
        )

        # Calculate final score with weighted average
        # Weights: dependencies and architecture (30%), run/validate/quality (70%)
        final_score = result_1.score * 0.30 + result_2.score * 0.70

        metadata = {
            "instructions": instructions,
            "semantic_test_1_capabilities_and_architecture": {
                "score": result_1.score,
                "details": result_1.metadata,
            },
            "semantic_test_2_run_validate_quality": {
                "score": result_2.score,
                "details": result_2.metadata,
            },
            "final_score": final_score,
            "scoring_weights": {
                "capabilities_and_architecture": "30%",
                "run_validate_quality": "70%",
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
