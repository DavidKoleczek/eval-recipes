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
NOTE: The solution may use either direct OpenAI API calls or an Agent SDK such as \
Claude Agent/Code SDK, Microsoft Amplifier (https://github.com/microsoft/amplifier/tree/next), OpenAI \
Codex CLI, or others that are similarly capable. Both approaches are acceptable for the image \
generation API calls."""

IMAGE_GEN_DOCS_URL = "https://platform.openai.com/docs/guides/image-generation"

# region Semantic Test 1: Architecture and Code Quality

STEPS_1_ARCHITECTURE = f"""\
{AGENT_SDK_NOTE}

1. First, look up the latest OpenAI image generation model by fetching {IMAGE_GEN_DOCS_URL}. \
Note the recommended model name for image generation.
2. Explore the code under scenarios/pixel_art_generator/ to understand the implementation.
3. Check that the code uses an OpenAI image generation model. Compare the model used against \
what you found in the docs -- is it the latest recommended model?
4. Check that the code requests a transparent background from the API so sprites can be used \
in games.
5. Check that the post-processing pipeline produces actual pixel art:
   - Resizing must avoid blurring (nearest-neighbor interpolation, not bilinear/bicubic/lanczos)
   - Colors are reduced to a limited palette
   - Output is a PNG with an alpha channel (transparency)
6. Verify the three CLI commands exist: generate, inspect, and batch, with the options \
described in the instructions (--size, --palette-size, --output, --output-dir).
7. Check for a README with usage examples."""

RUBRIC_1_ARCHITECTURE = {
    "uses_latest_model": (
        "str - (20 points) Does the code use the latest OpenAI image generation model "
        "from the docs? State which model the docs recommend and which model the code uses. "
        "Full points if it matches, 10 points partial credit if it uses an older OpenAI image model."
    ),
    "transparent_background": (
        "str - (15 points) Does the code request a transparent background from the API so sprites are game-ready?"
    ),
    "no_blurry_resize": (
        "str - (15 points) Does resizing use nearest-neighbor interpolation? "
        "Any smoothing/blurring interpolation method is wrong for pixel art."
    ),
    "color_reduction": "str - (15 points) Does the code reduce colors to a limited palette?",
    "png_with_alpha": "str - (10 points) Does the code output PNGs with an alpha channel for transparency?",
    "cli_commands": "str - (10 points) Are the three commands (generate, inspect, batch) present with the expected options?",
    "readme_exists": "str - (5 points) Does a README exist with usage examples?",
    "code_quality": "str - (10 points) Is the code well-organized with reasonable error handling?",
    "score": "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion.",
}

# endregion

# region Semantic Test 2: Functional Run

STEPS_2_FUNCTIONAL = """\
This test validates that the tool actually works end-to-end. A validation script is \
available at /project/test_time_data/validate_sprites.py that checks sprite properties.

1. Read the README under scenarios/pixel_art_generator/ to learn how to run the tool.
2. Install any required dependencies.
3. Create the /project/test_outputs/ directory.

4. Generate three sprites at different sizes:
   pixelart generate "a knight with a sword and shield" --size 32 --palette-size 16 \
--output /project/test_outputs/knight.png
   pixelart generate "a health potion bottle" --size 64 --palette-size 8 \
--output /project/test_outputs/potion.png
   pixelart generate "a treasure chest with gold coins" --size 16 --palette-size 12 \
--output /project/test_outputs/chest.png

5. Run the inspect command on one of the sprites and check the output looks reasonable.

6. Test batch mode. Create /project/test_outputs/batch_input.json with:
   [
     {"name": "tree", "description": "a pixel art oak tree"},
     {"name": "gem", "description": "a blue diamond gemstone"}
   ]
   Then run: pixelart batch /project/test_outputs/batch_input.json --size 32 \
--output-dir /project/test_outputs/batch/

7. Run the validation script on the generated sprites:
   uv run --with Pillow /project/test_time_data/validate_sprites.py \
/project/test_outputs/ --expected-size 32 --expected-palette 16
   Note: this will validate all PNGs recursively. The --expected-size and \
--expected-palette flags are global hints -- some sprites have different sizes, \
so focus on whether each sprite matches what was requested for it.

8. Read the validation output carefully. Check if sprites pass: dimensions, \
background removal (transparency), color palette limits, and anti-aliasing checks.

9. Visually assess: do the sprites look like pixel art you could use in a game?"""

RUBRIC_2_FUNCTIONAL = {
    "generate_works": "str - (15 points) Does the generate command run and produce PNG files?",
    "inspect_works": "str - (5 points) Does the inspect command show useful sprite info?",
    "correct_dimensions": "str - (15 points) Do sprites match their requested --size (32x32, 64x64, 16x16)?",
    "correct_palette": "str - (10 points) Are color counts within the requested --palette-size limits?",
    "has_transparency": "str - (10 points) Do sprites have transparent backgrounds (not white or solid)?",
    "no_antialiasing": "str - (5 points) Do sprites pass the anti-aliasing check from the validation script?",
    "batch_works": "str - (10 points) Does batch mode produce the expected sprite files?",
    "looks_like_pixel_art": (
        "str - (15 points) Do the sprites look like actual pixel art? Blocky, limited colors, no smooth gradients."
    ),
    "game_ready": (
        "str - (15 points) Could these sprites be dropped into a game? "
        "Transparent background, clean edges, recognizable subjects."
    ),
    "score": "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion.",
}

# endregion


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
    """Test script for pixel_art_generator task."""
    return asyncio.run(run_test(test_id, output_dir, instructions_file))


async def run_test(test_id: str, output_dir: Path, instructions_file: Path | None) -> int:
    instructions = get_instructions_from_file_or_default(instructions_file=instructions_file)

    try:
        logger.info("Running semantic test 1: Checking architecture and implementation...")
        result_1 = await semantic_test(
            steps=STEPS_1_ARCHITECTURE,
            rubric=RUBRIC_1_ARCHITECTURE,
            context=instructions,
            working_dir=Path("/project"),
        )

        logger.info("Running semantic test 2: Functional testing (generation, inspection, batch)...")
        result_2 = await semantic_test(
            steps=STEPS_2_FUNCTIONAL,
            rubric=RUBRIC_2_FUNCTIONAL,
            context=instructions,
            working_dir=Path("/project"),
        )

        # Weights: architecture (30%), functional (70%)
        final_score = result_1.score * 0.30 + result_2.score * 0.70

        metadata = {
            "instructions": instructions,
            "semantic_test_1_architecture": {
                "score": result_1.score,
                "details": result_1.metadata,
            },
            "semantic_test_2_functional": {
                "score": result_2.score,
                "details": result_2.metadata,
            },
            "final_score": final_score,
            "scoring_weights": {
                "architecture": "30%",
                "functional": "70%",
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
