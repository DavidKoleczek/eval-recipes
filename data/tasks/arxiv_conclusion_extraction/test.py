# Copyright (c) Microsoft. All rights reserved.

"""
Test script for arxiv paper conclusion extraction task.

Checks if Conclusion.txt contains the exact conclusion from https://arxiv.org/abs/2411.10541.
Scoring: 100 points for 0-5 character differences (buffer), -1 point per additional character difference (using Levenshtein distance).
"""

import json
import os
import sys
from pathlib import Path

GROUNDTRUTH_CONCLUSION = """Our study reveals that the way prompts are formatted significantly impacts GPT-based models' performance, with no single format excelling universally. This finding questions current evaluation methods that often ignore prompt structure, potentially misjudging a model's true abilities. We advocate for diverse prompt formats in future LLM testing to accurately gauge and enhance their performance.
Regarding explainability, we observe that model size affects model's responses to prompt variations. For instance, GPT-4's performance is less influenced by prompt changes compared to GPT-3.5, suggesting that larger models may process prompts more consistently. This discovery prompts further research into LLM interpretability, aiming to refine AI adaptability and human-AI interaction."""


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def main() -> int:
    # Check if Conclusion.txt exists
    conclusion_file = Path(__file__).parents[0] / "Conclusion.txt"
    if not conclusion_file.exists():
        print("FAIL: Conclusion.txt not found")
        test_id = os.environ["EVAL_RECIPES_TEST_ID"]
        result = {
            "score": 0,
            "metadata": {
                "groundtruth": GROUNDTRUTH_CONCLUSION,
                "llm_output": "",
            },
        }
        result_file = Path(__file__).parents[0] / f".eval_recipes_test_results_{test_id}.json"
        result_file.write_text(json.dumps(result))
        return 1

    llm_output = conclusion_file.read_text()
    edit_distance = levenshtein_distance(GROUNDTRUTH_CONCLUSION, llm_output)

    # Calculate score: start at 100, allow 5 character buffer, then subtract 1 for each additional difference
    score = max(0, 100 - max(0, edit_distance - 5))

    print(f"\nGroundtruth length: {len(GROUNDTRUTH_CONCLUSION)} characters")
    print(f"LLM output length: {len(llm_output)} characters")
    print(f"Edit distance: {edit_distance}")
    print(f"Score: {score:.1f}")

    test_id = os.environ["EVAL_RECIPES_TEST_ID"]
    result = {
        "score": score,
        "metadata": {
            "edit_distance": edit_distance,
            "groundtruth": GROUNDTRUTH_CONCLUSION,
            "llm_output": llm_output,
            "groundtruth_length": len(GROUNDTRUTH_CONCLUSION),
            "llm_output_length": len(llm_output),
        },
    }
    result_file = Path(__file__).parents[0] / f".eval_recipes_test_results_{test_id}.json"
    result_file.write_text(json.dumps(result))
    return 0 if edit_distance == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
