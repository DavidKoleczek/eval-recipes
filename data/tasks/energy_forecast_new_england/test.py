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

# Define Semantic Test 1: Data Acquisition and ML Implementation Review

STEPS_1_IMPLEMENTATION = """\
1. Explore the /project directory to understand the full implementation.
2. Look for dependency files (pyproject.toml, requirements.txt, etc.) to understand what ML libraries \
are used.
3. Check for data acquisition code:
   - Does it download real historical energy data from a public source?
   - What source does it use? Is it a legitimate public energy or government data source?
   - How much data does it acquire? (At least 30 days of hourly data = 720+ data points)
4. Check the ML implementation:
   - What algorithm is used? (Linear regression, random forest, gradient boosting, neural network, etc.)
   - Is there feature engineering beyond just raw demand values?
   - Look for time-based features (hour of day, day of week, month, holiday indicators)
   - Look for lagged features (previous hours/days demand)
   - Look for external features (temperature, weather data)
5. Check for model validation:
   - Is there a train/test split or cross-validation?
   - Are evaluation metrics computed (RMSE, MAE, MAPE, etc.)?
6. Check for a model_report.md that documents the approach.
7. Check for a README.md with usage instructions."""

RUBRIC_1_IMPLEMENTATION: dict[str, str] = {
    "real_data_source": """\
str - (15 points) Does the code download real historical energy data from a legitimate public source? \
Verify the data source URL or API is real. Full credit for a regional grid operator's published data. \
10 points for other legitimate energy or government data sources.""",
    "sufficient_data_volume": """\
str - (10 points) Does the code acquire at least 30 days of hourly data (720+ data points)? More data \
is better. 10 points for 30+ days, 5 points for 7-30 days, 0 for less.""",
    "ml_algorithm_quality": """\
str - (25 points) What ML algorithm is used? 25 points for gradient boosting, ensemble methods, or \
neural networks. 15 points for a single decision tree model or support vector regression. 5 points \
for linear regression or trivial approaches. 0 if no ML model.""",
    "feature_engineering": """\
str - (20 points) Does the code perform meaningful feature engineering? Check for: time-based features \
(hour, day of week, month) worth 8 points; lagged demand features worth 6 points; rolling statistics \
(moving averages, etc.) worth 3 points; external data like weather worth 3 points.""",
    "model_validation": """\
str - (10 points) Is there model validation? 10 points for cross-validation or robust train/test \
methodology with reported metrics. 5 points for a simple train/test split with metrics. 0 for no \
validation.""",
    "model_report_exists": """\
str - (10 points) Does model_report.md exist with a meaningful description of the data source, ML \
approach, feature engineering, and validation results?""",
    "readme_exists": """\
str - (10 points) Does a README.md exist with clear usage and installation instructions?""",
    "score": """\
float - Score between 0 and 100 based on the above criteria. Sum the points earned from each \
criterion.""",
}


# Define Semantic Test 2: Execution and Output Validation

STEPS_2_EXECUTION = """\
1. Find the README.md to understand how to run the tool.
2. Based on the README, install any required dependencies.
3. Run the tool. This may take up to 20 minutes as it needs to download data and train a model.
   - If the tool fails or errors out, note the error and assign a score of 0.
4. After completion, check that /project/forecast.json exists and is valid JSON.
5. Validate the forecast structure:
   - Does it contain a "forecast" array with 24 entries?
   - Does each entry have "hour" and "demand_mw" (numeric) fields?
   - Are the hours sequential (1-hour increments)?
6. Check that /project/model_report.md was generated and contains meaningful content.
7. Verify demand values are in a reasonable range for New England"""

RUBRIC_2_EXECUTION: dict[str, str] = {
    "tool_runs_successfully": """\
str - (30 points) Does the tool run without fatal errors and complete data download and model \
training?""",
    "forecast_json_exists": """\
str - (10 points) Does /project/forecast.json exist and contain valid JSON?""",
    "forecast_has_24_entries": """\
str - (15 points) Does the forecast array contain exactly 24 entries with sequential hourly \
timestamps?""",
    "forecast_entry_structure": """\
str - (10 points) Does each entry have 'hour' (ISO 8601 string) and 'demand_mw' (numeric) fields?""",
    "demand_values_reasonable_range": """\
str - (20 points) Are all 24 demand values within a plausible range for New England? Deduct points for values outside this range.""",
    "model_report_generated": """\
str - (15 points) Does /project/model_report.md exist with meaningful content about the approach and results?""",
    "score": """\
float - Score between 0 and 100 based on the above criteria. Sum the points earned from each \
criterion.""",
}


# Define Semantic Test 3: Forecast Plausibility Check

STEPS_3_PLAUSIBILITY = """\
1. Read the forecast from /project/forecast.json.
2. Determine what date range the forecast covers from the "hour" timestamps.
3. Try to look up current/recent New England electricity demand from public sources to establish a baseline \
for what demand should look like.
4. Look up current weather conditions and forecasts for the New England region to understand expected \
temperatures, which strongly influence electricity demand.
5. Analyze the forecast for temporal patterns:
   - Does demand show expected daily patterns (lower at night around 2-5 AM, higher during day)?
   - Is there appropriate variation between hours (not flat or random)?
   - Do the peak and trough values align with typical New England patterns?"""

RUBRIC_3_PLAUSIBILITY: dict[str, str] = {
    "daily_demand_pattern": """\
str - (35 points) Does the forecast show a realistic daily demand curve? Demand should be lower during \
nighttime hours (roughly 2-5 AM) and higher during daytime/evening. 35 points if the pattern is \
clearly visible and realistic. 20 points if there is some variation but the pattern is weak. 0 if \
demand is flat, random, or inversely patterned.""",
    "hour_to_hour_variation": """\
str - (25 points) Is there reasonable hour-to-hour variation (not all the same value, not wildly \
jumping)? Adjacent hours should typically differ by less than 1,000 MW. 25 points for smooth, \
realistic transitions. 12 points for somewhat reasonable. 0 for flat or erratic.""",
    "weather_consistency": """\
str - (20 points) Given current weather conditions and temperatures for the New England region, do \
the forecast demand levels make sense? Higher temperatures in summer or lower temperatures in winter \
should correspond to higher demand. 20 points if consistent with current conditions. 10 points if \
partially consistent. 0 if clearly inconsistent.""",
    "overall_forecast_credibility": """\
str - (20 points) Overall assessment: would an energy analyst find this forecast credible as a rough \
next-day forecast? Consider all the above factors holistically.""",
    "score": """\
float - Score between 0 and 100 based on the above criteria. Sum the points earned from each \
criterion.""",
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
    """Test script for energy_forecast_new_england task."""
    return asyncio.run(run_test(test_id, output_dir, instructions_file))


async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    try:
        logger.info("Running semantic test 1: Data acquisition and ML implementation review...")
        result_1 = await semantic_test(
            steps=STEPS_1_IMPLEMENTATION,
            rubric=RUBRIC_1_IMPLEMENTATION,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info("Running semantic test 2: Execution and output validation...")
        result_2 = await semantic_test(
            steps=STEPS_2_EXECUTION,
            rubric=RUBRIC_2_EXECUTION,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info("Running semantic test 3: Forecast plausibility check...")
        result_3 = await semantic_test(
            steps=STEPS_3_PLAUSIBILITY,
            rubric=RUBRIC_3_PLAUSIBILITY,
            context=instructions,
            working_dir=Path("/project"),
        )

        # Weights: implementation (25%), execution (35%), plausibility (40%)
        final_score = result_1.score * 0.25 + result_2.score * 0.35 + result_3.score * 0.40

        metadata = {
            "instructions": instructions,
            "semantic_test_1_implementation": {
                "score": result_1.score,
                "details": result_1.metadata,
            },
            "semantic_test_2_execution": {
                "score": result_2.score,
                "details": result_2.metadata,
            },
            "semantic_test_3_plausibility": {
                "score": result_3.score,
                "details": result_3.metadata,
            },
            "final_score": final_score,
            "scoring_weights": {
                "implementation": "25%",
                "execution": "35%",
                "plausibility": "40%",
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
