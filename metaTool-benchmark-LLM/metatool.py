#!/usr/bin/env python3
"""A small, local implementation of the MetaTool benchmark ideas.

The original project evaluates LLM answers. This module supplies the surrounding
benchmark machinery and a deterministic lexical baseline, so the benchmark can
be understood and run without a model server or vector database.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

TOKEN_RE = re.compile(r"[a-z0-9]+")
NO_TOOL_WORDS = {"write", "explain", "brainstorm", "opinion", "hello", "define"}
STOP_WORDS = {
    "a", "an", "and", "are", "for", "from", "how", "i", "in", "is", "it",
    "of", "on", "or", "the", "to", "what", "with", "you",
}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str


@dataclass(frozen=True)
class Candidate:
    tool: Tool
    score: float


@dataclass(frozen=True)
class Decision:
    use_tool: bool
    tools: tuple[str, ...]
    scores: tuple[tuple[str, float], ...]


def _tokens(text: str) -> list[str]:
    tokens = []
    for token in TOKEN_RE.findall(text.lower()):
        if token in STOP_WORDS:
            continue
        # A tiny normalization keeps "calculate" and "calculations" comparable
        # without requiring an NLP dependency.
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        elif token.endswith("ing") and len(token) > 5:
            token = token[:-3]
        if token.endswith("ion") and len(token) > 6:
            token = token[:-3]
        elif token.endswith("or") and len(token) > 6:
            token = token[:-2]
        elif token.endswith("e") and len(token) > 5:
            token = token[:-1]
        tokens.append(token)
    return tokens


def load_tools(path: str | Path) -> list[Tool]:
    """Load a JSON array, an OpenAI-style tool list, or a name-to-description map."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("tools", data)
        if isinstance(data, dict):
            data = [{"name": name, "description": description} for name, description in data.items()]
    if not isinstance(data, list):
        raise ValueError("tool catalog must be a JSON array or object")
    tools: list[Tool] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("each tool must be an object")
        function = item.get("function", item)
        name = function.get("name") or item.get("name")
        description = function.get("description") or item.get("description")
        if not isinstance(name, str) or not isinstance(description, str):
            raise ValueError("each tool needs string name and description")
        tools.append(Tool(name=name, description=description))
    if not tools:
        raise ValueError("tool catalog is empty")
    if len({tool.name for tool in tools}) != len(tools):
        raise ValueError("tool names must be unique")
    return tools


class LexicalSelector:
    """A transparent baseline for tool retrieval and selection.

    It uses TF-IDF-weighted token overlap rather than pretending to reproduce an
    LLM result. This makes errors inspectable and provides a useful control
    baseline for the paper's model-based experiments.
    """

    def __init__(self, tools: Sequence[Tool]) -> None:
        if not tools:
            raise ValueError("at least one tool is required")
        self.tools = list(tools)
        documents = [_tokens(f"{tool.name} {tool.description}") for tool in self.tools]
        document_frequency = Counter(token for document in documents for token in set(document))
        self.idf = {
            token: math.log((1 + len(documents)) / (1 + frequency)) + 1
            for token, frequency in document_frequency.items()
        }

    def rank(self, query: str, candidates: Sequence[Tool] | None = None) -> list[Candidate]:
        candidates = list(candidates or self.tools)
        query_tokens = Counter(_tokens(query))
        ranked: list[Candidate] = []
        for tool in candidates:
            tool_tokens = Counter(_tokens(f"{tool.name} {tool.description}"))
            overlap = set(query_tokens) & set(tool_tokens)
            score = sum(
                self.idf.get(token, 1.0) * min(query_tokens[token], tool_tokens[token])
                for token in overlap
            )
            # A tool name match is stronger than a description-only match.
            name_tokens = set(_tokens(tool.name))
            score += 1.5 * sum(self.idf.get(token, 1.0) for token in overlap & name_tokens)
            denominator = math.sqrt(sum(self.idf.get(token, 1.0) ** 2 for token in query_tokens))
            ranked.append(Candidate(tool, score / denominator if denominator else 0.0))
        return sorted(ranked, key=lambda item: (-item.score, item.tool.name))

    def decide(
        self, query: str, candidates: Sequence[Tool] | None = None, max_tools: int = 1,
        threshold: float = 0.28,
    ) -> Decision:
        if max_tools < 1:
            raise ValueError("max_tools must be positive")
        ranked = self.rank(query, candidates)
        query_tokens = set(_tokens(query))
        no_tool = bool(query_tokens & NO_TOOL_WORDS) and ranked[0].score < threshold
        selected = tuple(
            item.tool.name for item in ranked[:max_tools]
            if item.score >= threshold and not no_tool
        )
        return Decision(bool(selected), selected, tuple((item.tool.name, round(item.score, 6)) for item in ranked))


def build_prompt(query: str, tools: Sequence[Tool], multi: bool = False) -> str:
    """Build the paper's action-selection prompt without requiring LangChain."""
    limit = "You can choose two tools at most." if multi else "Choose one tool."
    descriptions = "\n".join(
        f"{index}. tool name: {tool.name}, tool description: {tool.description}"
        for index, tool in enumerate(tools, 1)
    )
    return (
        "You are a helpful AI assistant. Your current task is to choose the appropriate "
        "tool to solve the user's query.\n\n"
        f"User's Query:\n{query}\n\nList of Tools:\n{descriptions}\n\n"
        f"{limit} If no tool is applicable, return 'None'. Explain your choice briefly."
    )


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    data = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("data", data.get("records", [data]))
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("dataset must be a JSON array, JSONL-like array, or CSV")
    return data


def evaluate(selector: LexicalSelector, records: Iterable[dict[str, Any]], top_k: int = 1) -> dict[str, Any]:
    """Evaluate awareness and single/multi-tool selection accuracy.

    Accepted gold fields are ``tool`` (string), ``tools`` (list), and ``use_tool``
    (boolean). A record may also contain ``task`` for per-subtask metrics.
    """
    rows = list(records)
    awareness_correct = selection_correct = 0
    top_k_correct = 0
    by_task: dict[str, dict[str, int]] = {}
    for row in rows:
        query = str(row.get("query", ""))
        gold_tools = row.get("tools", row.get("tool", []))
        if isinstance(gold_tools, str):
            gold_tools = [] if gold_tools.lower() in {"none", "null", ""} else [gold_tools]
        gold = tuple(str(item) for item in gold_tools)
        expected_use = bool(row.get("use_tool", bool(gold)))
        decision = selector.decide(query, max_tools=max(1, len(gold)))
        awareness_correct += decision.use_tool == expected_use
        selection_correct += tuple(decision.tools) == gold
        top_k_correct += bool(set(gold) <= {name for name, _ in decision.scores[:top_k]})
        task = str(row.get("task", "all"))
        stats = by_task.setdefault(task, {"total": 0, "awareness_correct": 0, "selection_correct": 0})
        stats["total"] += 1
        stats["awareness_correct"] += decision.use_tool == expected_use
        stats["selection_correct"] += tuple(decision.tools) == gold
    total = len(rows)
    return {
        "total": total,
        "awareness_accuracy": awareness_correct / total if total else 0.0,
        "selection_accuracy": selection_correct / total if total else 0.0,
        "top_k_recall": top_k_correct / total if total else 0.0,
        "by_task": by_task,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a local MetaTool tool-selection baseline.")
    parser.add_argument("--tools", default="tools.json", help="tool catalog JSON")
    parser.add_argument("--query", help="query to classify")
    parser.add_argument("--dataset", help="JSON or CSV dataset to evaluate")
    parser.add_argument("--prompt", action="store_true", help="print the model-selection prompt")
    parser.add_argument("--max-tools", type=int, default=1)
    args = parser.parse_args(argv)
    selector = LexicalSelector(load_tools(args.tools))
    if args.dataset:
        print(json.dumps(evaluate(selector, _read_records(args.dataset), args.max_tools), indent=2))
        return 0
    if not args.query:
        parser.error("--query or --dataset is required")
    tools = selector.tools
    if args.prompt:
        print(build_prompt(args.query, tools, args.max_tools > 1))
    print(json.dumps(asdict(selector.decide(args.query, max_tools=args.max_tools)), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
