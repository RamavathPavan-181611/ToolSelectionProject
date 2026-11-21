#!/usr/bin/env python3
"""Self-Debugging LLM Tool Agent.

Combines:
- MetaTool: tool-awareness and tool-selection decisions
- AgentDebug: fine-grained failure analysis and root-cause correction
- GTA: explicit tool chains and end-to-end answer reporting
"""

from __future__ import annotations

import ast
import html
import json
import operator
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen

from llm_provider import create_provider


@dataclass(frozen=True)
class Tool:
    name: str
    category: str
    description: str
    parameters: tuple[str, ...]


@dataclass
class ToolCall:
    step: int
    name: str
    arguments: dict
    result: str = ""
    success: bool = False
    error: str = ""


@dataclass
class ModuleError:
    module: str
    error_type: str
    detected: bool
    evidence: str
    correction: str


@dataclass
class AgentReport:
    query: str
    use_tool: bool
    selected_tools: list[str]
    plan: list[str]
    tool_calls: list[dict]
    errors: list[dict]
    critical_error: dict | None
    answer: str
    recovered: bool
    provider: dict


TOOLS = [
    Tool("calculator", "logic", "Safely evaluate arithmetic expressions.", ("expression",)),
    Tool("file_reader", "operation", "Read a UTF-8 text file inside the project workspace.", ("path",)),
    Tool("text_statistics", "logic", "Count words, lines, characters, and sentences in text.", ("text",)),
    Tool("count_lines", "logic", "Count lines in text.", ("text",)),
    Tool("web_search", "perception", "Search the public web for current information.", ("query",)),
    Tool("image_inspector", "perception", "Inspect a local image format, dimensions, and file metadata.", ("path",)),
    Tool("ocr_image", "perception", "Extract readable text from a local image using local OCR.", ("path",)),
]


def safe_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class ToolSelector:
    """Confidence-scored hybrid intent router.

    It combines direct regex intent rules, phrase aliases, argument hints,
    and tool-description similarity. It is more capable than a fixed keyword
    dispatcher but remains deterministic and fast.
    """

    def __init__(self, tools: Iterable[Tool] = TOOLS) -> None:
        self.tools = list(tools)

    def rank(self, query: str) -> list[tuple[str, float]]:
        lowered = query.lower()
        scores = {tool.name: 0.0 for tool in self.tools}

        intent_rules = {
            "calculator": (
                0.95,
                r"\b(calculate|arithmetic|multiply|times|product|sum|subtract|divide)\b",
            ),
            "file_reader": (
                0.95,
                r"\b(read|open)\b.{0,40}\b(file|document|text)\b|\b(read|open)\b\s+[^\s]+\.(txt|md|csv|json|log)\b",
            ),
            "count_lines": (
                0.90,
                r"\b(count|number\s+of)\b.{0,20}\b(lines?|line\s+count)\b",
            ),
            "text_statistics": (
                0.88,
                r"\b(count|statistic|analy[sz]e|summar[y|ise])\b.{0,30}\b(words?|characters?|sentences?|text)\b",
            ),
            "web_search": (
                0.95,
                r"\b(search|look\s+up|find\s+online|browse|latest|current\s+news)\b",
            ),
            "ocr_image": (
                0.98,
                r"\b(ocr|extract\s+text|read\s+text)\b.{0,30}\b(image|picture|photo|screenshot)\b",
            ),
            "image_inspector": (
                0.96,
                r"\b(inspect|analy[sz]e|describe|dimensions?|metadata)\b.{0,30}\b(image|picture|photo|screenshot)\b",
            ),
        }
        for name, (weight, pattern) in intent_rules.items():
            if name in scores and re.search(pattern, lowered):
                scores[name] = max(scores[name], weight)

        if re.search(r"\b\d+(?:\.\d+)?\s*[+\-*\/%]\s*\d+(?:\.\d+)?\b", lowered):
            scores["calculator"] = max(scores["calculator"], 1.0)
        if re.search(r"\.(png|jpg|jpeg|heic|heif|tiff?|bmp|gif|webp)\b", lowered):
            for name in ("image_inspector", "ocr_image"):
                if name in scores:
                    scores[name] = max(scores[name], 0.82)

        query_tokens = safe_tokens(query)
        for tool in self.tools:
            tool_tokens = safe_tokens(f"{tool.name} {tool.description}")
            overlap = query_tokens & tool_tokens
            score = len(overlap) / max(1, sqrt(len(query_tokens) * len(tool_tokens)))
            scores[tool.name] = max(scores[tool.name], round(score * 0.55, 4))

        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))

    def confidence(self, query: str) -> dict[str, float]:
        return dict(self.rank(query))

    def decide(self, query: str, max_tools: int = 2) -> tuple[bool, list[str]]:
        ranked = self.rank(query)
        lowered = query.lower()

        no_tool_markers = (
            "explain", "brainstorm", "opinion", "hello", "define",
            "what is", "who is", "tell me",
        )
        has_explicit_tool_intent = any(
            phrase in lowered
            for phrase in (
                "calculate", "read ", "open ", "count ", "search ",
                "look up", "ocr ", "extract text", "inspect image",
                "analyze image", "image ", "picture ", "photo ", "screenshot ",
                "times ", "multiply", "plus ", "subtract", "divide",
            )
        )
        has_explicit_tool_intent = has_explicit_tool_intent or bool(
            re.search(r"\b\d+(?:\.\d+)?\s*[+\-*\/%]\s*\d+(?:\.\d+)?\b", lowered)
        )
        if any(marker in lowered for marker in no_tool_markers) and not has_explicit_tool_intent:
            if not re.search(r"\b\d+\s*[+\-*\/%]\s*\d+\b", lowered):
                return False, []

        selected = []
        for name, score in ranked:
            if score >= 0.76:
                selected.append(name)
        if not selected:
            strong = [(name, score) for name, score in ranked if score >= 0.30]
            selected = [name for name, _ in strong[:max_tools]]
        if "calculator" in selected:
            selected = ["calculator"]
        elif re.search(r"\b(inspect|analy[sz]e|describe|dimensions?|metadata)\b.{0,30}\b(image|picture|photo|screenshot)\b", lowered):
            selected = ["image_inspector"]
        elif "ocr_image" in selected:
            selected = ["ocr_image"]
        elif "image_inspector" in selected:
            selected = ["image_inspector"]
        elif "file_reader" in selected:
            selected = ["file_reader"]
        elif "count_lines" in selected:
            selected = ["count_lines"]
        elif "text_statistics" in selected:
            selected = ["text_statistics"]
        elif "web_search" in selected:
            selected = ["web_search"]
        return bool(selected), selected[:max_tools]


class SafeCalculator:
    """Evaluate arithmetic expressions without arbitrary Python execution."""

    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def run(self, expression: str) -> str:
        if len(expression) > 500:
            raise ValueError("expression is too long")

        def evaluate(node: ast.AST) -> float | int:
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in self.OPERATORS:
                left = evaluate(node.left)
                right = evaluate(node.right)
                if isinstance(node.op, ast.Pow) and abs(left) > 1000:
                    raise ValueError("unsafe exponent")
                return self.OPERATORS[type(node.op)](left, right)
            if isinstance(node, ast.UnaryOp) and type(node.op) in self.OPERATORS:
                return self.OPERATORS[type(node.op)](evaluate(node.operand))
            raise ValueError("only arithmetic expressions are allowed")

        value = evaluate(ast.parse(expression, mode="eval"))
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return str(value)


def text_statistics(text: str) -> str:
    lines = text.splitlines()
    words = re.findall(r"\b[\w'-]+\b", text)
    sentences = [item for item in re.split(r"[.!?]+", text) if item.strip()]
    return json.dumps({
        "characters": len(text),
        "words": len(words),
        "lines": len(lines),
        "sentences": len(sentences),
    })


class WebSearch:
    """Search the public web through DuckDuckGo HTML, with bounded results."""

    def run(self, query: str, limit: int = 5, timeout: float = 10.0) -> str:
        query = re.sub(r"\s+", " ", query).strip()
        if not query:
            raise ValueError("search query is required")
        if len(query) > 500:
            raise ValueError("search query is too long")
        limit = max(1, min(limit, 10))

        request = Request(
            "https://html.duckduckgo.com/html/?q=" + quote_plus(query),
            headers={
                "User-Agent": "AgentShield/1.0 (+local research project)",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urlopen(request, timeout=timeout) as response:
            body = response.read(4096 * 1024).decode("utf-8", "replace")

        results = []
        blocks = re.findall(
            r'<div class="result[^"]*">(.*?)</div>\s*(?:<div|$)',
            body,
            flags=re.I | re.S,
        )
        for block in blocks:
            link_match = re.search(
                r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                block,
                flags=re.I | re.S,
            )
            if not link_match:
                continue
            raw_url = html.unescape(link_match.group(1))
            title = re.sub(r"<[^>]+>", "", html.unescape(link_match.group(2))).strip()
            parsed = urlparse(raw_url)
            destination = raw_url
            if parsed.query:
                uddg = parse_qs(parsed.query).get("uddg")
                if uddg:
                    destination = unquote(uddg[0])

            snippet_match = re.search(
                r'<(?:a|div)[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>',
                block,
                flags=re.I | re.S,
            )
            snippet = ""
            if snippet_match:
                snippet = re.sub(r"<[^>]+>", " ", html.unescape(snippet_match.group(1)))
                snippet = re.sub(r"\s+", " ", snippet).strip()

            if title and destination.startswith(("http://", "https://")):
                results.append({
                    "title": title,
                    "url": destination,
                    "snippet": snippet,
                })
            if len(results) >= limit:
                break

        if not results:
            instant_request = Request(
                "https://api.duckduckgo.com/?"
                + "q=" + quote_plus(query)
                + "&format=json&no_html=1&skip_disambig=1",
                headers={"User-Agent": "AgentShield/1.0 (+local research project)"},
            )
            with urlopen(instant_request, timeout=timeout) as response:
                instant = json.loads(response.read().decode("utf-8", "replace"))

            instant_items = []
            if instant.get("AbstractURL") and instant.get("Heading"):
                instant_items.append({
                    "title": instant["Heading"],
                    "url": instant["AbstractURL"],
                    "snippet": instant.get("AbstractText", ""),
                })
            instant_items.extend(
                {
                    "title": item.get("Text", "Related result").split(" - ", 1)[0][:200],
                    "url": item.get("FirstURL", ""),
                    "snippet": item.get("Text", ""),
                }
                for item in instant.get("Results", [])
                if item.get("FirstURL")
            )
            for topic in instant.get("RelatedTopics", []):
                if "Topics" in topic:
                    instant_items.extend(
                        {
                            "title": item.get("Text", "Related result").split(" - ", 1)[0][:200],
                            "url": item.get("FirstURL", ""),
                            "snippet": item.get("Text", ""),
                        }
                        for item in topic["Topics"]
                        if item.get("FirstURL")
                    )
                elif topic.get("FirstURL"):
                    instant_items.append({
                        "title": topic.get("Text", "Related result").split(" - ", 1)[0][:200],
                        "url": topic["FirstURL"],
                        "snippet": topic.get("Text", ""),
                    })
            results.extend(
                item for item in instant_items
                if item["url"].startswith(("http://", "https://"))
            )
            results = results[:limit]

        if not results:
            wiki_request = Request(
                "https://en.wikipedia.org/w/api.php?"
                + "action=query&generator=search&gsrlimit="
                + str(limit)
                + "&gsrsearch=" + quote_plus(query)
                + "&prop=extracts|info&inprop=url&exintro=1&explaintext=1&format=json",
                headers={"User-Agent": "AgentShield/1.0 (local research project)"},
            )
            with urlopen(wiki_request, timeout=timeout) as response:
                wiki = json.loads(response.read().decode("utf-8", "replace"))
            pages = sorted(
                wiki.get("query", {}).get("pages", {}).values(),
                key=lambda page: page.get("index", 10**9),
            )
            results.extend({
                "title": page.get("title", ""),
                "url": page.get("fullurl", ""),
                "snippet": page.get("extract", "")[:500],
            } for page in pages if page.get("fullurl"))
            results = results[:limit]

        if not results:
            raise RuntimeError("No public web results were returned")
        return json.dumps(results)


class WorkspaceFileReader:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    def run(self, path: str) -> str:
        target = (self.workspace / path).resolve()
        if target != self.workspace and self.workspace not in target.parents:
            raise ValueError("path is outside the workspace")
        if not target.is_file():
            raise FileNotFoundError(path)
        if target.stat().st_size > 2_000_000:
            raise ValueError("file exceeds 2 MB")
        return target.read_text(encoding="utf-8")


class MultimodalImageTools:
    ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".heic", ".heif", ".tif", ".tiff", ".bmp", ".gif", ".webp"}

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    def _resolve(self, path: str) -> Path:
        target = (self.workspace / path).resolve()
        if target != self.workspace and self.workspace not in target.parents:
            raise ValueError("image path is outside the workspace")
        if not target.is_file():
            raise FileNotFoundError(path)
        if target.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            raise ValueError("unsupported image format")
        if target.stat().st_size > 10_000_000:
            raise ValueError("image exceeds 10 MB")
        return target

    def inspect(self, path: str) -> str:
        target = self._resolve(path)
        completed = subprocess.run(
            ["sips", "-g", "pixelWidth", "-g", "pixelHeight", "-g", "format", str(target)],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "macOS sips could not inspect image")
        metadata = {}
        for line in completed.stdout.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key in {"pixelWidth", "pixelHeight", "format"}:
                    metadata[key] = value
        metadata.update({
            "path": str(target.relative_to(self.workspace)),
            "bytes": target.stat().st_size,
            "extension": target.suffix.lower(),
        })
        return json.dumps(metadata)

    def ocr(self, path: str) -> str:
        target = self._resolve(path)
        tesseract = shutil.which("tesseract")
        if not tesseract:
            raise RuntimeError("Tesseract OCR is not installed")
        completed = subprocess.run(
            [tesseract, str(target), "stdout", "--psm", "6", "-l", "eng"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "OCR could not process image")
        text = re.sub(r"\n{3,}", "\n\n", completed.stdout).strip()
        if not text:
            return json.dumps({"path": str(target.relative_to(self.workspace)), "text": "", "note": "No text detected"})
        return json.dumps({
            "path": str(target.relative_to(self.workspace)),
            "text": text,
        })


class SelfDebuggingAgent:
    """End-to-end agent with MetaTool selection and AgentDebug diagnosis."""

    def __init__(
        self,
        workspace: Path | None = None,
        provider: str = "llama_cpp",
        model: str | None = None,
    ) -> None:
        self.workspace = (workspace or Path.cwd()).resolve()
        self.llm = create_provider(provider, model)
        self.provider_status = self.llm.status()
        self.selector = ToolSelector()
        self.calculator = SafeCalculator()
        self.file_reader = WorkspaceFileReader(self.workspace)
        self.web_search = WebSearch()
        self.image_tools = MultimodalImageTools(self.workspace)
        self.tools: dict[str, Callable[[dict], str]] = {
            "calculator": lambda args: self.calculator.run(str(args.get("expression", ""))),
            "file_reader": lambda args: self.file_reader.run(str(args.get("path", ""))),
            "text_statistics": lambda args: text_statistics(str(args.get("text", ""))),
            "count_lines": lambda args: str(len(str(args.get("text", "")).splitlines())),
            "web_search": lambda args: self.web_search.run(str(args.get("query", ""))),
            "image_inspector": lambda args: self.image_tools.inspect(str(args.get("path", ""))),
            "ocr_image": lambda args: self.image_tools.ocr(str(args.get("path", ""))),
        }

    def plan(self, query: str, selected: list[str]) -> list[str]:
        if not selected:
            return ["Answer directly from the model or supplied context."]

        plan = []
        for name in selected:
            if name == "calculator":
                plan.append("Extract the arithmetic expression.")
                plan.append("Validate and execute it with the calculator.")
            elif name == "file_reader":
                plan.append("Validate the requested workspace file path.")
                plan.append("Read and inspect the text file.")
            elif name == "text_statistics":
                plan.append("Collect the supplied text.")
                plan.append("Calculate counts.")
            elif name == "count_lines":
                plan.append("Collect the supplied text.")
                plan.append("Count text lines.")
            elif name == "web_search":
                plan.append("Extract the public-web search query.")
                plan.append("Retrieve and rank bounded search results.")
            elif name == "image_inspector":
                plan.append("Validate the local image path.")
                plan.append("Read dimensions and image format.")
            elif name == "ocr_image":
                plan.append("Validate the local image path.")
                plan.append("Extract readable text with local OCR.")
        plan.append("Combine tool observations into the final answer.")
        return plan

    def extract_expression(self, query: str) -> str:
        match = re.search(r"(-?\d+(?:\.\d+)?(?:\s*[+\-*\/%]\s*-?\d+(?:\.\d+)?)+)", query)
        if match:
            return match.group(1).replace("^", "**")
        number_words = {
            "zero": "0", "one": "1", "two": "2", "three": "3",
            "four": "4", "five": "5", "six": "6", "seven": "7",
            "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
            "twelve": "12", "thirteen": "13", "fourteen": "14",
            "fifteen": "15", "twenty": "20", "hundred": "100",
        }
        words = re.findall(r"[a-z]+", query.lower())
        translated = [number_words.get(word) for word in words]
        operands = [value for value in translated if value is not None]
        if len(operands) >= 2:
            if re.search(r"\b(times|multiply|multiplied)\b", query, re.I):
                return f"{operands[0]} * {operands[1]}"
            if re.search(r"\b(plus|add|sum)\b", query, re.I):
                return f"{operands[0]} + {operands[1]}"
            if re.search(r"\b(minus|subtract)\b", query, re.I):
                return f"{operands[0]} - {operands[1]}"
            if re.search(r"\b(divide|divided)\b", query, re.I):
                return f"{operands[0]} / {operands[1]}"
        words = re.findall(r"[-+]?\d+(?:\.\d+)?", query)
        if len(words) >= 2 and re.search(r"\b(sum|add|plus)\b", query, re.I):
            return " + ".join(words)
        if len(words) >= 2 and re.search(r"\b(product|multiply|times)\b", query, re.I):
            return " * ".join(words)
        return ""

    def build_calls(self, query: str, selected: list[str], supplied_text: str = "") -> list[ToolCall]:
        calls = []
        for index, name in enumerate(selected, 1):
            arguments: dict = {}
            if name == "calculator":
                arguments["expression"] = self.extract_expression(query)
            elif name == "file_reader":
                match = re.search(r"(?:read|open|file)\s+([^\s,;]+)", query, re.I)
                arguments["path"] = match.group(1) if match else ""
            elif name in {"text_statistics", "count_lines"}:
                arguments["text"] = supplied_text
            elif name == "web_search":
                search_query = re.sub(
                    r"^\s*(?:search(?:\s+the\s+web)?|look\s+up|find\s+online|current\s+news|latest|find)\b"
                    r"\s*(?:for|about)?\s*",
                    "",
                    query,
                    flags=re.I,
                ).strip()
                if search_query.lower() in {"the web", "web", "for", "about"}:
                    search_query = ""
                arguments["query"] = search_query
            elif name == "image_inspector":
                match = re.search(
                    r"(?:^|\s)(uploads?/[^\s,;]+|[A-Za-z0-9._/-]+\.(?:png|jpe?g|heic|heif|tiff?|bmp|gif|webp))",
                    query,
                    re.I,
                )
                arguments["path"] = match.group(1).strip(".,") if match else ""
            elif name == "ocr_image":
                match = re.search(
                    r"(?:^|\s)(uploads?/[^\s,;]+|[A-Za-z0-9._/-]+\.(?:png|jpe?g|heic|heif|tiff?|bmp|gif|webp))",
                    query,
                    re.I,
                )
                arguments["path"] = match.group(1).strip(".,") if match else ""
            calls.append(ToolCall(index, name, arguments))
        return calls

    def diagnose(self, call: ToolCall) -> ModuleError | None:
        if call.success:
            return None
        if call.name == "calculator" and not call.arguments.get("expression"):
            return ModuleError(
                "action", "parameter_error", True,
                "The calculator expression could not be extracted.",
                "Extract a valid arithmetic expression and include it in the parameters.",
            )
        if call.name == "file_reader" and not call.arguments.get("path"):
            return ModuleError(
                "action", "parameter_error", True,
                "The file path parameter is missing.",
                "Ask the user for a workspace-relative file path.",
            )
        if call.name == "web_search" and not call.arguments.get("query"):
            return ModuleError(
                "action", "parameter_error", True,
                "The public-web search query is missing.",
                "Ask the user for a specific search topic.",
            )
        if call.name in {"image_inspector", "ocr_image"} and not call.arguments.get("path"):
            return ModuleError(
                "action", "parameter_error", True,
                "The local image path is missing.",
                "Upload the image or provide a workspace-relative image path.",
            )
        if call.error:
            return ModuleError(
                "system", "tool_execution_error", True,
                call.error,
                "Handle the tool error, verify its inputs, and retry safely.",
            )
        return None

    def execute(self, query: str, supplied_text: str = "", max_retries: int = 1) -> AgentReport:
        use_tool, selected = self.selector.decide(query)
        raw_calls = [asdict(call) for call in self.build_calls(query, selected, supplied_text)]
        plan = self.plan(query, selected)
        calls = [
            ToolCall(index, raw["name"], raw["arguments"])
            for index, raw in enumerate(raw_calls, 1)
            if isinstance(raw, dict) and "name" in raw and "arguments" in raw
        ]
        errors: list[dict] = []

        for call in calls:
            for attempt in range(max_retries + 1):
                try:
                    if not call.arguments or any(value == "" for value in call.arguments.values()):
                        raise ValueError("missing or empty required tool arguments")
                    call.result = self.tools[call.name](call.arguments)
                    call.success = True
                    break
                except Exception as exc:
                    call.error = str(exc)
                    call.success = False
                    diagnosis = self.diagnose(call)
                    if diagnosis:
                        diagnostic = asdict(diagnosis)
                        if diagnostic not in errors:
                            errors.append(diagnostic)
                    if attempt == max_retries:
                        break

        critical_error = None
        if errors:
            critical_error = errors[0]

        if calls:
            successful = [call for call in calls if call.success]
            if successful:
                answer = "\n".join(
                    f"{call.name}: {call.result}" for call in successful
                )
                recovered = bool(critical_error is not None)
            else:
                answer = "The request could not be completed. " + (
                    critical_error["correction"] if critical_error else "Review the tool inputs."
                )
                recovered = False
        elif use_tool:
            answer = "No applicable tool call could be constructed."
            recovered = False
        else:
            if self.provider_status.available:
                try:
                    answer = self.llm.generate(
                        query,
                        system="Answer briefly and accurately. Do not claim that tools were used.",
                    )
                except Exception as exc:
                    answer = f"No external tool is required, but the provider failed: {exc}"
            else:
                answer = "No external tool is required. The response can be answered directly."
            recovered = False

        return AgentReport(
            query=query,
            use_tool=use_tool,
            selected_tools=selected,
            plan=plan,
            tool_calls=[asdict(call) for call in calls],
            errors=errors,
            critical_error=critical_error,
            answer=answer,
            recovered=recovered,
            provider=asdict(self.provider_status),
        )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the self-debugging tool agent")
    parser.add_argument("query", nargs="*", help="task for the agent")
    parser.add_argument("--text", default="", help="optional supplied text")
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--provider", choices=("llama_cpp", "ollama", "gemini"), default="llama_cpp")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()
    query = " ".join(args.query).strip()
    if not query:
        parser.error("a query is required")
    from llm_provider import OllamaProvider
    if args.base_url:
        agent = SelfDebuggingAgent(Path(args.workspace), args.provider, args.model)
        if isinstance(agent.llm, OllamaProvider):
            agent.llm = OllamaProvider(args.model or "llama3.2:3b", args.base_url)
            agent.provider_status = agent.llm.status()
    else:
        agent = SelfDebuggingAgent(Path(args.workspace), args.provider, args.model)
    report = agent.execute(query, args.text)
    print(json.dumps(asdict(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
