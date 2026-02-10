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
its built-in LLM capabilities count as satisfying the LLM usage checks for each stage below."""

STEPS_1_LLM_PIPELINE = f"""\
{AGENT_SDK_NOTE}

1. Explore the code under scenarios/email_drafting/ to understand the full implementation.
2. Check that it reads past emails from a directory and source notes/bullet points from a file.
3. Check for a **writer** stage that uses an LLM to draft an email based on the notes and past email style.
4. Check for a **content-reviewer** stage that uses an LLM to verify all key points from the notes are covered \
and nothing important was missed or misrepresented.
5. Check for a **tone-reviewer** stage that uses an LLM to verify the draft matches the user's communication \
style, formality level, and typical email patterns from past emails.
6. Check for a **brevity-reviewer** stage that uses an LLM to verify email length is appropriate for the \
selected mode and is not unnecessarily verbose.
7. Check for revision logic: if any reviewer finds issues, the draft should go back to the writer and through \
the reviewers again.
8. Check for a mode flag (--mode concise|standard|detailed) that controls email length.
9. Check for an optional recipient parameter that prioritizes past emails to/from that specific recipient.
10. Check for a feedback loop: the user can mark up the draft with [bracket-enclosed-comments] and pass it \
back to the tool to incorporate feedback, restarting the write-and-review cycle.
11. Verify the solution uses a recent, up-to-date LLM from Anthropic or OpenAI. If an Agent SDK is used, \
check which model it is configured with. Check the model identifier in the code against these references:
    - Anthropic models: https://platform.claude.com/docs/en/about-claude/models/overview
    - OpenAI models: https://platform.openai.com/docs/models"""

RUBRIC_1_LLM_PIPELINE = {
    "reads_past_emails": "str - (5 points) Does the code read past emails from a directory?",
    "reads_source_notes": "str - (5 points) Does the code read source notes/bullet points from a file?",
    "writer_uses_llm": "str - (10 points) Is there a writer stage that uses an LLM to draft an email matching the user's style? Agent SDKs with built-in LLM capabilities count.",
    "content_reviewer_uses_llm": "str - (10 points) Is there a content-reviewer stage that uses an LLM to verify key points are covered? Agent SDKs with built-in LLM capabilities count.",
    "tone_reviewer_uses_llm": "str - (10 points) Is there a tone-reviewer stage that uses an LLM to verify the draft matches communication style? Agent SDKs with built-in LLM capabilities count.",
    "brevity_reviewer_uses_llm": "str - (5 points) Is there a brevity-reviewer stage that uses an LLM to verify appropriate length for the selected mode? Agent SDKs with built-in LLM capabilities count.",
    "revision_loop": "str - (10 points) If a reviewer finds issues, does the draft go back to the writer and through reviewers again?",
    "mode_flag": "str - (5 points) Is there a mode flag (concise/standard/detailed) controlling email length?",
    "recipient_parameter": "str - (5 points) Is there an optional recipient parameter that prioritizes matching past emails?",
    "feedback_loop": "str - (5 points) Can the user mark up the draft with [bracket-enclosed-comments] and pass it back?",
    "uses_recent_model": "str - (25 points) Does it use a recent model from Anthropic (see https://platform.claude.com/docs/en/about-claude/models/overview) or OpenAI (see https://platform.openai.com/docs/models)? If an Agent SDK is used, check which model it is configured with. 5 points partial credit if a model is used but it is not recent.",
    "score": "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion.",
}

# Define Semantic Test 2

STEPS_2_USE_TOOL = """1. Find the README under scenarios/email_drafting/ that the agent should have made to explain how the use the tool and the commands to run.
2. Based on the README, you should come up with the appropriate command to test the tool with the following inputs:
  - Use the set of sample emails available in the `emails` directory 
  - Use the source_notes.txt file that contains the notes for the new email
  - Use "sarah.chen" as the recipient.
3. Run the command to generate an email draft. Depending on how its implemented, this may take up to 10 minutes.
4. After the command finishes, first check if it wrote the email to a file.
5. If it did, did the content of the email match \
  a) the style of the previous emails in the emails dir \
  b) Was it also more aligned to previous emails written to Sara Chen?"""

RUBRIC_2_USE_TOOL = {
    "readme_exists": "str - (5 points) did you find a README file and it contained correct instructions",
    "correct tool inputs": "str - (15 points) did the command have all of the correct inputs: emails dir, notes file, recipient name",
    "tool ran without error": "str - (30 points) did the tool run without error",
    "tool produced file version": "str - (20 points) did the tool produce an email file that could be marked up with [bracket-enclosed-comments]?",
    "email file style": "str - (10 points) did the tool create an email that generally aligns with the style of previous emails?",
    "email recipient style": "str - (20 points) did the tool create an email that generally aligns with the style of previous emails written to Sara Chen?",
    "score": "float - Score between 0 and 100 based on the above criteria.",
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
    """Test script for email_drafting task."""
    return asyncio.run(run_test(test_id, output_dir, instructions_file))


async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    try:
        logger.info("Running semantic test 1 to verify LLM pipeline implementation...")
        result_1 = await semantic_test(
            steps=STEPS_1_LLM_PIPELINE,
            rubric=RUBRIC_1_LLM_PIPELINE,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info("Running semantic test 2 to check if the tool works as intended...")
        result_2 = await semantic_test(
            steps=STEPS_2_USE_TOOL,
            rubric=RUBRIC_2_USE_TOOL,
            context=instructions,
            working_dir=Path("/project"),
        )

        final_score = (result_1.score + result_2.score) / 2
        metadata = {
            "instructions": instructions,
            "semantic_test_1_score": result_1.score,
            "semantic_test_1_metadata": result_1.metadata,
            "semantic_test_2_score": result_2.score,
            "semantic_test_2_metadata": result_2.metadata,
            "final_score": final_score,
        }

        write_test_result(output_dir, test_id, final_score, metadata)
        return 0

    except Exception as e:
        metadata = {
            "instructions": instructions,
            "error": str(e),
        }
        write_test_result(output_dir, test_id, 0, metadata)
        return 0


if __name__ == "__main__":
    sys.exit(main())
