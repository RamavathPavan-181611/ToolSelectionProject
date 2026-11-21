import json
import tempfile
import unittest
from pathlib import Path

from metatool import LexicalSelector, Tool, build_prompt, evaluate, load_tools


class MetaToolTests(unittest.TestCase):
    def setUp(self):
        self.tools = [
            Tool("weather", "Get current weather and forecasts."),
            Tool("calculator", "Perform arithmetic calculations."),
            Tool("writer", "Write and edit prose."),
        ]
        self.selector = LexicalSelector(self.tools)

    def test_rank_prefers_relevant_tool(self):
        self.assertEqual(self.selector.decide("What is the weather in Paris?").tools, ("weather",))
        self.assertEqual(self.selector.decide("Calculate 2 plus 2.").tools, ("calculator",))

    def test_no_tool_decision_is_supported(self):
        decision = self.selector.decide("Explain recursion in simple terms.")
        self.assertFalse(decision.use_tool)
        self.assertEqual(decision.tools, ())

    def test_multi_tool_and_prompt(self):
        decision = self.selector.decide("Check the weather and calculate the temperature average.", max_tools=2)
        self.assertEqual(set(decision.tools), {"weather", "calculator"})
        prompt = build_prompt("Find the forecast.", self.tools)
        self.assertIn("tool name: weather", prompt)
        self.assertIn("Find the forecast.", prompt)

    def test_evaluate_reports_awareness_and_selection(self):
        result = evaluate(self.selector, [
            {"query": "weather tomorrow", "tool": "weather", "task": "scenario"},
            {"query": "Explain recursion", "use_tool": False, "tool": "None", "task": "awareness"},
        ])
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["awareness_accuracy"], 1.0)
        self.assertEqual(result["selection_accuracy"], 1.0)
        self.assertEqual(result["by_task"]["scenario"]["total"], 1)

    def test_load_openai_style_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tools.json"
            path.write_text(json.dumps({"tools": [
                {"type": "function", "function": {"name": "search", "description": "Search web"}}
            ]}))
            self.assertEqual(load_tools(path), [Tool("search", "Search web")])


if __name__ == "__main__":
    unittest.main()
