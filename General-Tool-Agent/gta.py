#!/usr/bin/env python3
"""Local GTA-1 benchmark implementation.

Implements the paper's dataset format and evaluation calculations:
- step-by-step: InstAcc, ToolAcc, ArgAcc, SummAcc
- end-to-end: AnsAcc and Perception/Operation/Logic/Creativity tool F1

The official 229-task dataset is included under official_gta_dataset/.
There are no remote tool executions or model API calls in this baseline.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable


CATEGORIES = {
    "P": {"OCR", "CountGivenObject", "ImageDescription", "RegionAttributeDescription",
          "TextToBbox", "MathOCR", "GoogleSearch"},
    "O": {"Plot", "DrawBox", "AddText"},
    "L": {"Calculator", "Solver"},
    "C": {"TextToImage", "ImageStylization"},
}

TOOL_CATEGORY = {
    tool: category
    for category, tools in CATEGORIES.items()
    for tool in tools
}


def load_gta_dataset(path: str | Path) -> dict[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("GTA dataset.json must contain task IDs mapped to task objects")
    return data


def ground_truth_steps(task: dict) -> list[dict]:
    steps = []
    for dialog in task.get("dialogs", []):
        for call in dialog.get("tool_calls", []):
            function = call.get("function", call)
            steps.append({
                "name": function.get("name", ""),
                "arguments": function.get("arguments", {}),
            })
    return steps


def ground_truth_tools(task: dict) -> set[str]:
    return {step["name"] for step in ground_truth_steps(task)}


def get_user_query(task: dict) -> str:
    for dialog in task.get("dialogs", []):
        if dialog.get("role") == "user":
            return str(dialog.get("content", "")).strip()
    return ""


def get_ground_truth_answer(task: dict) -> str:
    for dialog in reversed(task.get("dialogs", [])):
        if dialog.get("role") == "assistant" and not dialog.get("tool_calls"):
            content = dialog.get("content")
            if isinstance(content, str):
                return content
    gt_answer = task.get("gt_answer", {})
    whitelist = gt_answer.get("whitelist", []) if isinstance(gt_answer, dict) else []
    if whitelist:
        first = whitelist[0]
        return first if isinstance(first, str) else str(first)
    return ""


def inst_accuracy(predictions: list[dict], gold: list[dict]) -> float:
    if not gold:
        return 0.0
    correct = sum(
        prediction.get("valid_format", True) is True
        and isinstance(prediction.get("name"), str)
        for prediction in predictions[:len(gold)]
    )
    correct += sum(
        1 for _ in []  # no extra steps are treated as N/A below
    )
    return correct / len(gold)


def tool_accuracy(predictions: list[dict], gold: list[dict]) -> float:
    if not gold:
        return 0.0
    correct = sum(
        prediction.get("name") == expected["name"]
        for prediction, expected in zip(predictions, gold)
    )
    return correct / len(gold)


def argument_accuracy(predictions: list[dict], gold: list[dict]) -> float:
    if not gold:
        return 0.0
    correct = 0
    for prediction, expected in zip(predictions, gold):
        if prediction.get("name") != expected["name"]:
            continue
        predicted_arguments = prediction.get("arguments", {})
        expected_arguments = expected["arguments"]
        if isinstance(predicted_arguments, dict) and predicted_arguments == expected_arguments:
            correct += 1
    return correct / len(gold)


def normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def answer_correct(prediction: str, gt_answer: dict | str | None) -> bool:
    if gt_answer is None:
        return False
    if isinstance(gt_answer, str):
        return normalize_answer(prediction) == normalize_answer(gt_answer)
    if not isinstance(gt_answer, dict):
        return normalize_answer(prediction) == normalize_answer(gt_answer)
    whitelist = gt_answer.get("whitelist")
    blacklist = gt_answer.get("blacklist")
    predicted = normalize_answer(prediction)
    if whitelist:
        alternatives = whitelist
        if isinstance(alternatives, list) and alternatives and isinstance(alternatives[0], list):
            alternatives = alternatives[0]
        if not isinstance(alternatives, list):
            alternatives = [alternatives]
        if not any(normalize_answer(str(item)) in predicted for item in alternatives):
            return False
    if blacklist:
        for item in blacklist:
            if normalize_answer(str(item)) in predicted:
                return False
    return bool(predicted)


def summary_accuracy(predictions: list[dict], gold_answer: str) -> float:
    if not predictions:
        return 0.0
    summary = str(predictions[-1].get("summary", "")).strip()
    if not summary:
        return 0.0
    return float(normalize_answer(summary) == normalize_answer(gold_answer))


def f1_for_category(predicted: Iterable[str], gold: Iterable[str], category: str) -> float:
    category_tools = CATEGORIES[category]
    predicted_set = {name for name in predicted if name in category_tools}
    gold_set = {name for name in gold if name in category_tools}
    if not predicted_set and not gold_set:
        return 1.0
    tp = len(predicted_set & gold_set)
    fp = len(predicted_set - gold_set)
    fn = len(gold_set - predicted_set)
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else 0.0


def evaluate_step_by_step(predictions_by_task: dict[str, list[dict]], dataset: dict[str, dict]) -> dict:
    inst = tool = arg = summary = 0.0
    count = 0
    for task_id, task in dataset.items():
        gold = ground_truth_steps(task)
        predicted = predictions_by_task.get(task_id, [])
        inst += inst_accuracy(predicted, gold)
        tool += tool_accuracy(predicted, gold)
        arg += argument_accuracy(predicted, gold)
        summary += summary_accuracy(predicted, get_ground_truth_answer(task))
        count += 1
    return {
        "InstAcc": inst / count if count else 0.0,
        "ToolAcc": tool / count if count else 0.0,
        "ArgAcc": arg / count if count else 0.0,
        "SummAcc": summary / count if count else 0.0,
    }


def evaluate_end_to_end(predictions_by_task: dict[str, dict], dataset: dict[str, dict]) -> dict:
    answer_correct_count = 0
    text_task_count = 0
    category_scores = {category: [] for category in CATEGORIES}
    for task_id, task in dataset.items():
        prediction = predictions_by_task.get(task_id, {})
        predicted_answer = str(prediction.get("answer", ""))
        gt_answer = task.get("gt_answer")
        if gt_answer:
            text_task_count += 1
            answer_correct_count += answer_correct(predicted_answer, gt_answer)
        predicted_tools = set(prediction.get("tools", []))
        gold_tools = ground_truth_tools(task)
        for category in CATEGORIES:
            category_scores[category].append(
                f1_for_category(predicted_tools, gold_tools, category)
            )
    return {
        "AnsAcc": answer_correct_count / text_task_count if text_task_count else 0.0,
        "P": sum(category_scores["P"]) / len(category_scores["P"]) if category_scores["P"] else 0.0,
        "O": sum(category_scores["O"]) / len(category_scores["O"]) if category_scores["O"] else 0.0,
        "L": sum(category_scores["L"]) / len(category_scores["L"]) if category_scores["L"] else 0.0,
        "C": sum(category_scores["C"]) / len(category_scores["C"]) if category_scores["C"] else 0.0,
    }


def replay_ground_truth(dataset: dict[str, dict]) -> tuple[dict, dict]:
    """Create prediction records from the official ground-truth chain."""
    step_predictions = {}
    end_predictions = {}
    for task_id, task in dataset.items():
        gold = ground_truth_steps(task)
        step_predictions[task_id] = [
            {"name": step["name"], "arguments": step["arguments"],
             "valid_format": True, "summary": get_ground_truth_answer(task)}
            for step in gold
        ]
        end_predictions[task_id] = {
            "answer": get_ground_truth_answer(task),
            "tools": [step["name"] for step in gold],
        }
    return step_predictions, end_predictions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the GTA-1 local benchmark utility")
    parser.add_argument("--dataset", default="official_gta_dataset/dataset.json")
    parser.add_argument("--task", help="print one official task and its ground-truth chain")
    parser.add_argument("--validate", action="store_true",
                        help="replay official ground-truth chains as a metric sanity check")
    args = parser.parse_args(argv)
    dataset = load_gta_dataset(args.dataset)
    if args.task:
        task = dataset.get(args.task)
        if task is None:
            parser.error(f"unknown task ID: {args.task}")
        print(json.dumps({
            "task_id": args.task,
            "query": get_user_query(task),
            "images": task.get("files", []),
            "tools": [tool["name"] for tool in task.get("tools", [])],
            "ground_truth_steps": ground_truth_steps(task),
            "ground_truth_answer": get_ground_truth_answer(task),
        }, indent=2))
        return 0
    if args.validate:
        step_predictions, end_predictions = replay_ground_truth(dataset)
        print(json.dumps({
            "tasks": len(dataset),
            "ground_truth_tool_calls": sum(len(steps) for steps in step_predictions.values()),
            "step_by_step": evaluate_step_by_step(step_predictions, dataset),
            "end_to_end": evaluate_end_to_end(end_predictions, dataset),
        }, indent=2))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
