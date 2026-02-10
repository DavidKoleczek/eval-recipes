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

STEPS = """\
1. Find the README.md file that explains how to use the tool and what commands to run.
2. Based on the README:
   - Install any required dependencies if not already installed.
   - Determine the correct command to run the tool with default settings.
3. Run the tool. This may take up to 15 minutes.
   - If the tool fails or errors out, note down an overall score of 0
4. After completion, locate the output CSV file and examine its structure:
   - Does it have headers matching the required columns (company_name, expected_date, exchange, \
sector, description, source_url)?
   - Is the CSV properly formatted (valid CSV syntax, parseable without errors)?
   - Does it contain at least 5 company entries?
5. Validate data quality in the CSV:
   - Are company names real companies (not placeholder or dummy data)?
   - Are expected dates reasonable (within the next 365 days from now)?
   - Are sector fields meaningful (e.g., "Technology", "Healthcare", not empty or generic)?
   - Are descriptions meaningful one-line summaries?
   - Are source URLs present and not empty?
6. Cross-validate companies against web searches:
   - Pick 3-5 companies from the CSV output.
   - For each company, search the web to independently verify that the company is actually planning or has filed for an IPO.
   - Check if the search results corroborate the IPO claim for each company.
   - A company is verified if you can find at least one credible source confirming its IPO plans.
7. Test source URL accessibility:
   - Pick 2-3 source_url values from the CSV.
   - Make HTTP requests to verify they are actually accessible (return 200/300-level responses)."""

RUBRIC = {
    "tool_runs_successfully": "str - (20 points) Does the tool run without errors?",
    "csv_created_with_columns": "str - (10 points) Is a CSV file created with the required columns and proper formatting?",
    "csv_has_enough_entries": "str - (10 points) Does the CSV contain at least 5 company entries?",
    "companies_verified_via_search": "str - (30 points) When cross-validating 3-5 companies from the output by searching the web, can you find credible sources confirming their IPO plans? Award full points if most companies are verified, partial credit if some are verified, 0 if none can be verified or companies appear fabricated.",
    "source_urls_accessible": "str - (10 points) When testing 2-3 source URLs with HTTP requests, are they accessible and relevant to the IPO claim?",
    "data_fields_meaningful": "str - (20 points) Are company descriptions, sectors, and dates meaningful and specific (not empty or generic)?",
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
    """Test script for ipo_tracker task."""
    return asyncio.run(run_test(test_id, output_dir, instructions_file))


async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    try:
        logger.info("Running semantic test: Running tool and validating output...")
        result = await semantic_test(
            steps=STEPS,
            rubric=RUBRIC,
            context=instructions,
            working_dir=Path("/project"),
        )

        metadata = {
            "instructions": instructions,
            "semantic_test_result": result.metadata,
        }

        write_test_result(output_dir, test_id, result.score, metadata)
        logger.info(f"Test completed with final score: {result.score:.1f}/100")
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
