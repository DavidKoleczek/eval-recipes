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
1. Find and read the README to understand how to run the CPSC recall tool.
2. Based on the README, install any required dependencies if needed.
3. Run the tool with "December 2024" (or "2024-12"). This may take several minutes as it needs to \
fetch data from CPSC.
   - If the tool fails to run or produces errors, note the specific error and assign a score of 0.
4. After the tool completes, locate the output CSV file and examine its structure:
   - Does it have headers?
   - Are the required columns present: product name, recall date, manufacturer, hazard description, \
and recall URL?
   - Is the CSV properly formatted (valid CSV syntax, parseable without errors)?
   - Does it contain recall records (not empty)?
5. Validate data quality:
   - Check that recall dates are actually from December 2024
   - Verify URLs are valid CPSC recall URLs (should contain "cpsc.gov" or similar official CPSC domain)
   - Check that manufacturer fields contain actual company names (not empty or generic)
   - Check that hazard descriptions are meaningful and specific
   - Check that product names are specific product descriptions
6. Test URL validity:
   - Pick 2-3 random URLs from the CSV
   - Make HTTP requests to verify they are actually accessible (return 200/300-level responses)
   - If accessible, spot-check that the information in the CSV matches what's on the CPSC website
7. Test alternative month format:
   - Run the tool again with "2024-11" (November 2024)
   - Verify it handles the different format correctly and outputs recalls from November 2024"""

RUBRIC = {
    "readme_clear": "str - (5 points) Is the README clear about how to run the tool?",
    "tool_runs_successfully": "str - (20 points) Does the tool run without errors when given 'December 2024'?",
    "csv_file_created": "str - (10 points) Is a CSV file created as output?",
    "required_columns_present": "str - (10 points) Are all required columns present (product name, recall date, manufacturer, hazard, URL) with proper headers?",
    "csv_properly_formatted": "str - (5 points) Is the CSV properly formatted and parseable?",
    "csv_has_data": "str - (5 points) Does the CSV contain recall records with critical fields populated?",
    "recall_dates_correct_month": "str - (10 points) Are the recall dates actually from the requested month (December 2024)?",
    "urls_are_valid_cpsc": "str - (5 points) Are the URLs valid CPSC recall URLs (proper format, contain cpsc.gov)?",
    "urls_actually_work": "str - (10 points) When testing 2-3 URLs with HTTP requests, are they accessible and does the CSV data match the CPSC pages?",
    "data_fields_meaningful": "str - (10 points) Are manufacturer names, hazard descriptions, and product names meaningful and specific (not empty or generic)?",
    "alternative_format_works": "str - (10 points) Does the tool work with alternative month format ('2024-11') and produce recalls from November 2024?",
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
    """Test script for cpsc_recall_monitor task."""
    return asyncio.run(run_test(test_id, output_dir, instructions_file))


async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    try:
        logger.info("Running semantic test: Validating tool output and data quality...")
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
