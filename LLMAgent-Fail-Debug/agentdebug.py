#!/usr/bin/env python3
"""Dependency-free local implementation of the AgentDebug paper's core pipeline.

This baseline follows the upstream AgentDebug shape:
trajectory parsing -> step/module fine-grained analysis -> critical root-cause
detection -> correction guidance. It is intentionally transparent: replacing
detect_module_errors/identify_critical_error with an LLM adapter preserves the
same JSON interface.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


TAXONOMY_PATH = Path(__file__).with_name("taxonomy.json")


@dataclass
class ModuleError:
    module_name: str
    error_type: str
    error_detected: bool
    evidence: str
    reasoning: str


@dataclass
class StepAnalysis:
    step: int
    memory_error: ModuleError | None
    reflection_error: ModuleError | None
    planning_error: ModuleError | None
    action_error: ModuleError | None
    system_error: ModuleError | None
    step_summary: str


@dataclass
class CriticalError:
    critical_step: int
    critical_module: str
    error_type: str
    root_cause: str
    evidence: str
    correction_guidance: str
    cascading_effects: list[dict]
    confidence: float


def load_taxonomy(path: str | Path = TAXONOMY_PATH) -> dict[str, dict[str, str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_trajectory(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    messages = data.get("messages", data.get("chat_history", []))
    metadata = data.get("metadata", {})
    task_description = data.get("task_description") or metadata.get("task", "")
    if not task_description:
        for message in messages:
            content = str(message.get("content", ""))
            if message.get("role") == "user" and content:
                task_description = content.split("\n", 1)[0][:500]
                break

    steps = []
    assistant_index = 0
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        assistant_index += 1
        env_response = ""
        if index + 1 < len(messages) and messages[index + 1].get("role") in {"user", "environment"}:
            env_response = str(messages[index + 1].get("content", ""))
        steps.append({
            "step": assistant_index,
            "content": str(message.get("content", "")),
            "env_response": env_response,
        })
    return {
        "task_id": metadata.get("task_id", "unknown"),
        "task_description": task_description,
        "success": bool(metadata.get("success", metadata.get("won", False))),
        "environment": metadata.get("environment", data.get("environment", "unknown")),
        "steps": steps,
        "total_steps": len(steps),
    }


def extract_modules(content: str, environment: str = "unknown") -> dict[str, str]:
    modules = {}
    tags = {"memory": "memory", "reflection": "reflection", "planning": "plan",
             "action": "action"}
    for module, tag in tags.items():
        match = re.search(rf"<{tag}>(.*?)</{tag}>", content, re.DOTALL | re.IGNORECASE)
        modules[module] = match.group(1).strip() if match else ""
    memory_match = re.search(r"<memory_recall>(.*?)</memory_recall>", content, re.DOTALL)
    if memory_match and not modules["memory"]:
        modules["memory"] = memory_match.group(1).strip()
    if environment.lower() == "webshop" and not modules["planning"]:
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if match:
            modules["planning"] = match.group(1).strip()
    return modules


def module_error(module: str, error_type: str, evidence: str, reasoning: str) -> ModuleError:
    return ModuleError(module, error_type, True, evidence, reasoning)


def no_error(module: str) -> ModuleError:
    return ModuleError(module, "no_error", False, "", "")


class HeuristicErrorDetector:
    """Transparent baseline for validating the AgentDebug pipeline."""

    def __init__(self, taxonomy: dict[str, dict[str, str]] | None = None) -> None:
        self.taxonomy = taxonomy or load_taxonomy()

    def analyze_trajectory(self, trajectory: dict) -> dict:
        step_analyses = []
        for step in trajectory["steps"]:
            modules = extract_modules(step["content"], trajectory["environment"])
            step_analyses.append(self.analyze_step(
                step["step"], step["content"], modules,
                step["env_response"], trajectory["environment"],
            ))
        return {
            "task_id": trajectory["task_id"],
            "task_description": trajectory["task_description"],
            "task_success": trajectory["success"],
            "environment": trajectory["environment"],
            "step_analyses": [asdict(item) for item in step_analyses],
        }

    def analyze_step(
        self, step_number: int, content: str, modules: dict[str, str],
        env_response: str, environment: str,
    ) -> StepAnalysis:
        memory = no_error("memory")
        reflection = no_error("reflection")
        planning = no_error("planning")
        action = no_error("action")
        system = no_error("system")

        if step_number > 1:
            memory_text = modules["memory"].lower()
            if re.search(r"\bremember(?:ed)?\b.*\b(?:never|did not|haven't)\b", memory_text):
                memory = module_error("memory", "hallucination", modules["memory"],
                                      "The agent recalls an event that is described as never happening.")
            elif re.search(r"\b(need|need to|find|locate)\b", memory_text) and re.search(
                r"\b(?:searched?|looked?)\b.*\b(?:wrong|other|another)\b", memory_text
            ):
                memory = module_error("memory", "memory_retrieval_failure", modules["memory"],
                                      "Relevant prior information was not retrieved during the decision.")
            elif len(memory_text.split()) > 35:
                memory = module_error("memory", "over_simplification", modules["memory"],
                                      "The memory summary is too broad and omits actionable detail.")

        reflection_text = modules["reflection"].lower()
        if re.search(r"\b(successfully|completed|done|finished)\b", reflection_text) and re.search(
            r"\bnothing happens\b|\bfailed\b|\bstill\b", env_response.lower()
        ):
            reflection = module_error("reflection", "outcome_misinterpretation", modules["reflection"],
                                      "The reflection claims success despite conflicting environment feedback.")
        elif re.search(r"\b(nearly complete|smooth|no progress|huge difficulties)\b", reflection_text):
            reflection = module_error("reflection", "progress_misjudge", modules["reflection"],
                                      "Progress is judged incorrectly relative to the task state.")
        elif re.search(r"\b(because|due to|probably|maybe)\b", reflection_text) and re.search(
            r"\b(mechanical|tool|network|model|malfunction)\b", reflection_text
        ):
            reflection = module_error("reflection", "causal_misattribution", modules["reflection"],
                                      "The reflection assigns failure to a cause weaker than the evidence.")
        elif re.search(r"\b(already|has been|completed)\b", reflection_text) and not re.search(
            r"\b(successfully|completed|done|finished)\b", env_response.lower()
        ):
            reflection = module_error("reflection", "hallucination", modules["reflection"],
                                      "The reflection claims a prior action completed without positive feedback.")

        planning_text = modules["planning"].lower()
        if re.search(r"\bbudget\b.{0,40}\b(over|exceed|ignore|\\$\\d+)\b", planning_text):
            planning = module_error("planning", "constraint_ignorance", modules["planning"],
                                    "The plan does not respect a stated budget or resource constraint.")
        elif re.search(r"\b(slice|put|move|open|use)\b", planning_text) and re.search(
            r"\b(no .*|empty|without|locked|impossible)\b", planning_text
        ):
            planning = module_error("planning", "impossible_action", modules["planning"],
                                    "The plan requires an unavailable prerequisite or impossible action.")
        elif re.search(r"\b(upstairs|bedroom|bathroom|back and forth|multiple)\b", planning_text):
            planning = module_error("planning", "inefficient_plan", modules["planning"],
                                    "The plan uses an avoidable, indirect, or redundant route.")

        action_text = modules["action"].strip()
        action_lower = action_text.lower()
        env_lower = env_response.lower()
        if action_text and re.search(r"(?:^|[<\\[{])[^\\]>\\]}]{0,5}$", action_text):
            action = module_error("action", "format_error", action_text,
                                  "The action has unmatched or malformed syntax.")
        elif re.search(r"\b(error|exception|crash|timeout|does not exist|unknown action)\b", env_lower):
            action = module_error("action", "invalid_action", action_text,
                                  "Environment feedback indicates an unknown action or execution failure.")
        elif re.search(r"\bgo ?to|take|search|put|open|click|buy\b", action_lower) and re.search(
            r"\bnot aligned|instead|contradict|unrelated\b", planning_text
        ):
            action = module_error("action", "misalignment", action_text,
                                  "The action choice contradicts the preceding plan.")
        elif re.search(r"\bgo ?to\b", action_lower) and re.search(
            r"\b(pick|take|put|open|search|slice|buy|click)\b", planning_text
        ):
            target_match = re.search(r"goto\(([^)]+)\)", action_lower)
            target = target_match.group(1).strip() if target_match else ""
            if not target or target not in planning_text:
                action = module_error("action", "misalignment", action_text,
                                      "The action only moves to a location while the plan specifies a task operation.")
        elif re.search(r"\[[^\\]]{80,}\\]|\b(-{1,}|repeat(?:ed)? 100)\b", action_text):
            action = module_error("action", "parameter_error", action_text,
                                  "The action parameter is unreasonable or malformed.")

        if re.search(r"\bstep limit\b|\bmaximum steps\b|\breached step\b", content.lower()):
            system = module_error("system", "step_limit", content,
                                  "The trajectory reached the system step limit.")
        elif re.search(r"\b(tool|api).*?(error|timeout|exception)|\bcrashed\b", env_lower):
            system = module_error("system", "environment_error", env_response,
                                  "The environment or external tool reported a crash or execution error.")
        elif re.search(r"\b(timeout|token limit|max tokens)\b", env_lower):
            system = module_error("system", "llm_limit", env_response,
                                  "The model or response limit caused the failure.")

        summary = "Fine-grained module analysis for step %d" % step_number
        return StepAnalysis(step_number, memory, reflection, planning, action, system, summary)

    def identify_critical_error(self, phase1: dict) -> CriticalError | None:
        if phase1["task_success"]:
            return None
        candidates = []
        for analysis in phase1["step_analyses"]:
            step = int(analysis["step"])
            for module in ("memory", "reflection", "planning", "action", "system"):
                error = analysis.get(module + "_error")
                if not error or not error.get("error_detected"):
                    continue
                if step == 1 and module in {"memory", "reflection"}:
                    continue
                candidates.append((step, module, error))
        if not candidates:
            return None
        step, module, error = min(candidates, key=lambda candidate: candidate[0])
        cascading = [
            {"step": later["step"], "effect": "Downstream module errors may propagate from the critical error"}
            for later in phase1["step_analyses"]
            if int(later["step"]) > step
            and any(
                later.get(name + "_error", {}).get("error_detected")
                for name in ("memory", "reflection", "planning", "action", "system")
            )
        ]
        return CriticalError(
            critical_step=step,
            critical_module=module,
            error_type=error["error_type"],
            root_cause=error["reasoning"],
            evidence=error["evidence"],
            correction_guidance=self.correction_guidance(module, error["error_type"]),
            cascading_effects=cascading,
            confidence=0.75,
        )

    def correction_guidance(self, module: str, error_type: str) -> str:
        guidance = {
            "memory": "Do not use unsupported recalled state; retrieve verified observations before acting.",
            "reflection": "Base progress judgments only on verified environment feedback.",
            "planning": "Construct a plan that respects prerequisites and task constraints.",
            "action": "Choose a valid action that directly implements the plan and use valid parameters.",
            "system": "Handle external failures explicitly, retry safely, or stop and report the system limit.",
        }
        return f"{guidance.get(module, 'Review the step and correct the invalid decision.')} "
        f"Address {error_type} rather than only fixing later symptoms."


def evaluate(results: Iterable[dict], gold_trajectories: Iterable[dict]) -> dict:
    predicted = list(results)
    gold = list(gold_trajectories)
    if len(predicted) != len(gold):
        raise ValueError("predicted and gold trajectory counts must match")
    step_correct = fine_total = critical_correct = all_correct = 0
    for phase1, gold_record in zip(predicted, gold):
        predicted_steps = phase1["step_analyses"]
        gold_steps = gold_record.get("step_analyses", [])
        for predicted_step, gold_step in zip(predicted_steps, gold_steps):
            for module in ("memory", "reflection", "planning", "action", "system"):
                fine_total += 1
                predicted_error = predicted_step[module + "_error"]
                gold_error = gold_step.get(module + "_error", {})
                if predicted_error["error_type"] == gold_error.get("error_type", "no_error"):
                    step_correct += 1

        predicted_critical = phase1.get("critical_error")
        gold_critical = gold_record.get("critical_error")
        if predicted_critical is not None and gold_critical is not None:
            predicted_tuple = (
                predicted_critical["critical_step"], predicted_critical["critical_module"],
                predicted_critical["error_type"],
            )
            gold_tuple = (
                gold_critical["critical_step"], gold_critical["critical_module"],
                gold_critical["error_type"],
            )
            critical_correct += predicted_critical["critical_step"] == gold_critical["critical_step"]
            all_correct += predicted_tuple == gold_tuple
        elif predicted_critical is None and gold_critical is None:
            all_correct += 1
    total = len(gold)
    return {
        "trajectories": total,
        "step_accuracy": step_correct / fine_total if fine_total else 0.0,
        "critical_step_accuracy": critical_correct / total if total else 0.0,
        "critical_all_correct_accuracy": all_correct / total if total else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the AgentDebug local debugging baseline")
    parser.add_argument("--trajectory", help="trajectory JSON file")
    parser.add_argument("--gold", help="gold JSON file for evaluation")
    parser.add_argument("--taxonomy", default=str(TAXONOMY_PATH))
    args = parser.parse_args(argv)
    detector = HeuristicErrorDetector(load_taxonomy(args.taxonomy))
    if args.trajectory:
        trajectory = parse_trajectory(args.trajectory)
        phase1 = detector.analyze_trajectory(trajectory)
        critical = detector.identify_critical_error(phase1)
        phase1["critical_error"] = asdict(critical) if critical else None
        print(json.dumps(phase1, indent=2))
        return 0
    if args.gold:
        gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
        gold = gold if isinstance(gold, list) else [gold]
        predicted = []
        for record in gold:
            phase1 = detector.analyze_trajectory(record)
            phase1["critical_error"] = asdict(detector.identify_critical_error(phase1))
            predicted.append(phase1)
        print(json.dumps(evaluate(predicted, gold), indent=2))
        return 0
    parser.error("--trajectory or --gold is required")
    return 2


if __name__ == "__main__":
    sys.exit(main())
