"""
Local evaluation script that mirrors the Kaggle competition metric.
Extracts answers from \boxed{} format and compares with ground truth.
"""

import re
import math
from typing import Optional


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} answer from model output."""
    # Find all \boxed{...} patterns (handling nested braces)
    matches = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        # Find matching closing brace
        depth = 0
        start = idx + len("\\boxed{")
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                if depth == 0:
                    matches.append(text[start:j])
                    i = j + 1
                    break
                depth -= 1
        else:
            break
        if not matches or i <= idx:
            i = idx + 1

    return matches[-1].strip() if matches else None


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    # Remove leading/trailing whitespace
    answer = answer.strip()
    # Remove enclosing $ signs
    if answer.startswith("$") and answer.endswith("$"):
        answer = answer[1:-1].strip()
    # Remove \text{} wrappers
    answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
    # Remove \mathrm{} wrappers
    answer = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", answer)
    return answer.strip()


def try_parse_number(s: str) -> Optional[float]:
    """Try to parse a string as a number."""
    s = s.strip().replace(",", "")
    # Handle fractions like \frac{a}{b}
    frac_match = re.match(r"\\frac\{([^}]+)\}\{([^}]+)\}", s)
    if frac_match:
        try:
            num = float(frac_match.group(1))
            den = float(frac_match.group(2))
            return num / den if den != 0 else None
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def answers_match(predicted: str, ground_truth: str, tolerance: float = 1e-2) -> bool:
    """Check if predicted answer matches ground truth.
    
    Uses exact string match first, then numerical tolerance of 1e-2.
    """
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    # Exact string match
    if pred_norm == gt_norm:
        return True

    # Numerical comparison with tolerance
    pred_num = try_parse_number(pred_norm)
    gt_num = try_parse_number(gt_norm)
    if pred_num is not None and gt_num is not None:
        return math.isclose(pred_num, gt_num, abs_tol=tolerance)

    return False


def evaluate_response(model_output: str, ground_truth: str) -> dict:
    """Evaluate a single model response against ground truth.
    
    Returns dict with 'correct', 'predicted', 'ground_truth', 'extracted'.
    """
    extracted = extract_boxed_answer(model_output)
    if extracted is None:
        return {
            "correct": False,
            "predicted": None,
            "ground_truth": ground_truth,
            "extracted": False,
        }

    correct = answers_match(extracted, ground_truth)
    return {
        "correct": correct,
        "predicted": extracted,
        "ground_truth": ground_truth,
        "extracted": True,
    }


if __name__ == "__main__":
    # Basic self-test
    tests = [
        ("The answer is \\boxed{42}", "42", True),
        ("Therefore \\boxed{3.14}", "3.14159", True),  # within 1e-2
        ("\\boxed{\\frac{1}{3}}", "0.333", True),
        ("No boxed answer here", "42", False),
        ("\\boxed{hello}", "world", False),
    ]
    for output, truth, expected in tests:
        result = evaluate_response(output, truth)
        status = "PASS" if result["correct"] == expected else "FAIL"
        print(f"[{status}] Output='{output[:40]}...' Truth='{truth}' -> {result['correct']}")
