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

# Define Semantic Test 1: Architecture and Implementation

AGENT_SDK_NOTE = """\
NOTE: The solution may use either direct LLM API calls (OpenAI, Anthropic) or an Agent SDK such as \
Claude Agent/Code SDK, Microsoft Amplifier (https://github.com/microsoft/amplifier/tree/next), OpenAI \
Codex CLI, or others that are similarly capable. Both approaches are acceptable. If an Agent SDK is used, \
its built-in LLM capabilities count as satisfying the LLM usage checks below."""

STEPS_1_ARCHITECTURE = f"""\
{AGENT_SDK_NOTE}

1. Explore the code under scenarios/chiptune_generator/ to understand the full implementation.
2. Check how the solution generates MIDI files. It should use a proper MIDI library (e.g. \
midiutil, mido, pretty_midi, or similar). Verify the library is listed in dependency files \
and actually imported and used in the code.
3. Check how the solution uses an LLM or Agent SDK to interpret natural language descriptions. \
There should be a reasonable separation of concerns - the LLM interprets the description into \
musical parameters, and separate logic translates those parameters into MIDI data. It should \
not be one giant function that does everything.
4. Check the MIDI generation code for chiptune-appropriate instrument choices:
   - Look for use of synth lead programs (MIDI programs 80-87 range covers square waves, \
sawtooth, etc.) or other chiptune-evocative sounds
   - Check if multiple channels are used (melody, bass, harmony, percussion)
   - The instrument choices should sound like chip tunes, not piano or orchestral instruments
5. Check for the three CLI commands the user requested:
   - A "generate" command that takes a description and produces a MIDI file
   - A "refine" command that takes an existing MIDI file and feedback to produce a modified file
   - An "analyze" command that prints a human-readable breakdown of a MIDI file
6. Check that the refine command actually reads and parses the existing MIDI file to understand \
its structure before modifying it. It should not simply regenerate from scratch while ignoring \
the input file.
7. Check for proper error handling: invalid files, missing API keys, bad descriptions.
8. Verify the solution uses a recent, up-to-date LLM from Anthropic or OpenAI. If an Agent SDK \
is used, check which model it is configured with. Check the model identifier in the code against \
these references:
    - Anthropic models: https://platform.claude.com/docs/en/about-claude/models/overview
    - OpenAI models: https://platform.openai.com/docs/models"""

RUBRIC_1_ARCHITECTURE = {
    "midi_library": (
        "str - (10 points) Does the solution use a proper MIDI library (midiutil, mido, "
        "pretty_midi, or similar) for generating MIDI files? Check both dependency files "
        "and actual imports/usage in code."
    ),
    "llm_midi_separation": (
        "str - (15 points) Is there a reasonable separation between LLM interpretation "
        "(producing structured musical parameters) and MIDI generation (consuming those "
        "parameters)? They should not be a single monolithic function."
    ),
    "chiptune_instruments": (
        "str - (15 points) Does the MIDI generation use chiptune-appropriate instruments? "
        "Look for synth lead programs (MIDI programs 80-87 for square/sawtooth waves), "
        "multiple channels (melody, bass, harmony, percussion), and instrument choices "
        "that evoke retro 8-bit sounds rather than piano or orchestral sounds."
    ),
    "three_cli_commands": ("str - (10 points) Are there three CLI commands: generate, refine, and analyze?"),
    "refine_reads_existing": (
        "str - (10 points) Does the refine command actually read and parse the existing "
        "MIDI file to understand its structure before modifying it (not generating from scratch)?"
    ),
    "error_handling": ("str - (5 points) Does the code handle errors properly (invalid files, missing API keys)?"),
    "readme_exists": (
        "str - (10 points) Does a README.md exist under scenarios/chiptune_generator/ "
        "with usage examples for all three commands (generate, refine, analyze)?"
    ),
    "uses_recent_model": (
        "str - (25 points) Does it use a recent model from Anthropic (see "
        "https://platform.claude.com/docs/en/about-claude/models/overview) or OpenAI "
        "(see https://platform.openai.com/docs/models)? If an Agent SDK is used, check "
        "which model it is configured with. 5 points partial credit if a model is used "
        "but it is not recent."
    ),
    "score": (
        "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion."
    ),
}


# Define Semantic Test 2: Functional -- Generation, Analysis, and Refinement

STEPS_2_FUNCTIONAL = """\
This test validates that the tool actually works by generating MIDI files, analyzing them, \
and testing refinement. Since you cannot listen to audio, you will use programmatic MIDI \
analysis as a proxy for quality.

1. Find the README under scenarios/chiptune_generator/ to understand how to run the tool.
2. Based on the README, install any required dependencies (pip install -r requirements.txt \
or similar).

--- GENERATION TEST ---

3. Run the generate command with this description:
   "a melancholy chiptune with a slow tempo, minor key, and a simple repeating melody"
   Save the output to /project/test_output/melancholy.mid
   Create the /project/test_output/ directory first if it does not exist.
4. Verify the MIDI file was created at the expected path.
5. Run the analyze command on /project/test_output/melancholy.mid:
   - Verify the analyze command runs without errors
   - Check that the output includes: duration, tempo (BPM), channels used, note counts, \
pitch ranges, velocities, and time signature

6. A pre-built MIDI analysis script is available at /project/analyze_midi.py.
   Install its dependency first: pip install mido
   Run it on the melancholy MIDI file:
   python /project/analyze_midi.py /project/test_output/melancholy.mid
   It will print a JSON object with fields: total_duration_seconds, tempo_bpm, num_tracks, \
channels_used, per_channel_note_count, per_channel_pitch_range, per_channel_avg_velocity, \
total_note_count, and programs_used.
7. Examine the JSON output from the analysis script. Verify:
   - The MIDI file has a duration between 30 and 120 seconds
   - The tempo is reasonable for a "slow melancholy" description (roughly 60-100 BPM)
   - At least 2 channels contain notes
   - At least one channel uses a chiptune-style MIDI program number (synth lead programs \
in the 80-87 range, or other retro-sounding instruments rather than piano/strings/brass)
   - Total note count is at least 50 (enough for a recognizable melody)
   - Notes show pitch and velocity variation (not all identical values)

--- CONTRASTING GENERATION TEST ---

8. Run generate a second time with a contrasting description:
   "a fast energetic 8-bit boss battle theme with rapid arpeggios and driving percussion"
   Save to /project/test_output/boss_battle.mid
9. Run the analysis script on boss_battle.mid and compare to melancholy.mid:
   python /project/analyze_midi.py /project/test_output/boss_battle.mid
   - The boss battle tune should have a higher tempo (BPM) than the melancholy tune
   - The boss battle tune should have higher note density (more total notes or shorter \
duration per note) than the melancholy tune
   - This verifies the LLM interpretation actually controls the musical output

--- REFINEMENT TEST ---

10. Generate a base tune:
    "a medium-tempo chiptune march with steady bass and a simple melody"
    Save to /project/test_output/march_v1.mid
11. Run the analysis script on march_v1.mid and save the JSON output for comparison:
    python /project/analyze_midi.py /project/test_output/march_v1.mid
12. Run the refine command on march_v1.mid with this feedback:
    "make the bass line much more prominent with higher velocity and add more bass notes"
    Save to /project/test_output/march_v2.mid
13. Run the analysis script on march_v2.mid and compare to march_v1.mid:
    python /project/analyze_midi.py /project/test_output/march_v2.mid
    - Identify the bass channel (the channel with the lowest average pitch range, or channel 3)
    - The bass channel should have HIGHER average velocity in v2 compared to v1
    - The bass channel should have MORE notes in v2 compared to v1
    - The overall duration should be similar (within 50% of the original)
14. Run the refine command on march_v1.mid with a different instruction:
    "double the tempo and make it much faster"
    Save to /project/test_output/march_v3.mid
15. Run the analysis script on march_v3.mid and compare to march_v1.mid:
    python /project/analyze_midi.py /project/test_output/march_v3.mid
    - The tempo in v3 should be noticeably higher than v1 (at least 30% faster)
    - The file should still be a valid MIDI with notes
16. Verify that refinement is not just regeneration: compare march_v1.mid and march_v2.mid. \
Non-bass channels should have similar note counts (within 50% of the original), indicating \
the refinement preserved aspects that were not asked to change."""

RUBRIC_2_FUNCTIONAL = {
    "generate_runs": (
        "str - (10 points) Does the generate command run without errors and produce "
        "a MIDI file for the melancholy description?"
    ),
    "midi_file_created": ("str - (5 points) Is a valid MIDI file created at the expected path?"),
    "analyze_works": (
        "str - (10 points) Does the analyze command run and produce a readable report "
        "with duration, tempo, channels, note counts, pitch ranges, and velocities?"
    ),
    "valid_duration": ("str - (5 points) Is the MIDI file duration between 30 and 120 seconds?"),
    "appropriate_tempo": (
        "str - (5 points) Is the tempo reasonable for the slow melancholy description (roughly 60-100 BPM)?"
    ),
    "multiple_channels": ("str - (5 points) Does the MIDI use at least 2 channels with notes?"),
    "chiptune_programs": (
        "str - (5 points) Does at least one channel use a chiptune-style MIDI program "
        "(synth lead programs in the 80-87 range, or other retro-sounding instruments "
        "rather than piano/strings/brass)?"
    ),
    "note_variety": ("str - (5 points) Do notes show pitch and velocity variation (not all identical)?"),
    "second_gen_different": (
        "str - (10 points) Is the boss battle tune measurably different from the melancholy "
        "tune? The boss battle should have higher tempo and/or higher note density."
    ),
    "refine_runs": (
        "str - (5 points) Does the refine command run without errors when given an existing "
        "MIDI file and modification instructions?"
    ),
    "bass_velocity_increased": (
        "str - (10 points) After requesting 'make the bass line more prominent with higher "
        "velocity', did the bass channel's average velocity increase compared to the original?"
    ),
    "bass_notes_increased": (
        "str - (10 points) After requesting 'add more bass notes', did the bass channel's "
        "note count increase compared to the original?"
    ),
    "tempo_refinement": (
        "str - (10 points) After requesting 'double the tempo', is the tempo in the refined "
        "version at least 30% faster than the original?"
    ),
    "refinement_preserves_structure": (
        "str - (5 points) Does refinement preserve aspects not asked to change? Non-bass "
        "channels should have similar note counts between the original and bass-refined version."
    ),
    "score": (
        "float - Score between 0 and 100 based on the above criteria. Sum the points earned from each criterion."
    ),
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
    """Test script for chiptune_generator task."""
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

        logger.info("Running semantic test 2: Functional testing (generation, analysis, refinement)...")
        result_2 = await semantic_test(
            steps=STEPS_2_FUNCTIONAL,
            rubric=RUBRIC_2_FUNCTIONAL,
            context=instructions,
            working_dir=Path("/project"),
        )

        # Weights: architecture (35%), functional (65%)
        final_score = result_1.score * 0.35 + result_2.score * 0.65

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
                "architecture": "35%",
                "functional": "65%",
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
