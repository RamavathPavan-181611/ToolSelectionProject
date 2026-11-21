import json
import tempfile
import unittest
from pathlib import Path

from agent import SelfDebuggingAgent, ToolSelector, SafeCalculator, WebSearch
from llm_provider import LlamaCppProvider, OllamaProvider, ProviderStatus
from memory_store import MemoryStore


class AgentShieldTests(unittest.TestCase):
    def setUp(self):
        self.agent = SelfDebuggingAgent(Path(__file__).resolve().parent)

    def test_meta_tool_awareness_and_selection(self):
        selector = ToolSelector()
        use_tool, tools = selector.decide("Calculate 12 * 8")
        self.assertTrue(use_tool)
        self.assertIn("calculator", tools)
        use_tool, tools = selector.decide("Explain what an API is")
        self.assertFalse(use_tool)
        self.assertEqual(tools, [])

    def test_file_request_does_not_select_other_tools(self):
        use_tool, tools = ToolSelector().decide("Read example.txt")
        self.assertTrue(use_tool)
        self.assertEqual(tools, ["file_reader"])

    def test_safe_calculator(self):
        calculator = SafeCalculator()
        self.assertEqual(calculator.run("12 * 8"), "96")
        self.assertEqual(calculator.run("(10 + 5) / 3"), "5")
        with self.assertRaises(ValueError):
            calculator.run("__import__('os').system('echo unsafe')")

    def test_calculator_end_to_end(self):
        report = self.agent.execute("Calculate 12 * 8")
        self.assertTrue(report.use_tool)
        self.assertIn("calculator", report.selected_tools)
        self.assertTrue(report.tool_calls[0]["success"])
        self.assertEqual(report.answer, "calculator: 96")

    def test_text_statistics(self):
        report = self.agent.execute(
            "Count words and sentences in this text",
            "Hello world. This is AgentShield.",
        )
        success = [call for call in report.tool_calls if call["success"]]
        self.assertTrue(success)
        stats = json.loads(success[0]["result"])
        self.assertEqual(stats["words"], 5)
        self.assertEqual(stats["sentences"], 2)

    def test_missing_calculator_argument_produces_diagnostic(self):
        report = self.agent.execute("Calculate this value")
        self.assertFalse(report.tool_calls[0]["success"])
        self.assertIsNotNone(report.critical_error)
        self.assertEqual(report.critical_error["module"], "action")
        self.assertEqual(report.critical_error["error_type"], "parameter_error")

    def test_file_reader_rejects_paths_outside_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            outside = Path(directory) / "secret.txt"
            outside.write_text("secret")
            agent = SelfDebuggingAgent(Path(__file__).resolve().parent)
            report = agent.execute(f"Read {outside}")
            self.assertTrue(report.tool_calls)
            self.assertFalse(report.tool_calls[0]["success"])

    def test_provider_status_is_exposed_in_report(self):
        report = self.agent.execute("Explain what an API is")
        self.assertIn(report.provider["provider"], {"llama_cpp", "ollama", "gemini"})
        self.assertIn("available", report.provider)
        self.assertIn("model", report.provider)

    def test_ollama_status_is_safe_when_service_is_down(self):
        provider = OllamaProvider(
            model="llama3.2:3b",
            base_url="http://127.0.0.1:1",
            timeout=1,
        )
        status = provider.status()
        self.assertFalse(status.available)

    def test_llama_cpp_provider_is_available(self):
        status = LlamaCppProvider().status()
        self.assertTrue(status.available)

    def test_natural_language_arithmetic_selects_calculator(self):
        report = self.agent.execute("What is twelve times eight?")
        self.assertTrue(report.use_tool)
        self.assertEqual(report.selected_tools, ["calculator"])
        self.assertEqual(report.answer, "calculator: 96")

    def test_image_routing_and_inspection(self):
        use_tool, tools = ToolSelector().decide("Inspect image uploads/sample.png")
        self.assertTrue(use_tool)
        self.assertEqual(tools, ["image_inspector"])
        with tempfile.TemporaryDirectory() as directory:
            import base64
            workspace = Path(directory)
            image = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAG0lEQVR4nGP4"
                "z8AARAyGg4eB4qEGJAASBQGJMwGQAAAAAElFTkSuQmCC"
            )
            (workspace / "sample.png").write_bytes(image)
            agent = SelfDebuggingAgent(workspace)
            report = agent.execute("Inspect image sample.png")
            self.assertTrue(report.tool_calls[0]["success"])

    def test_missing_image_path_is_diagnosed(self):
        report = self.agent.execute("OCR an image")
        self.assertEqual(report.selected_tools, ["ocr_image"])
        self.assertFalse(report.tool_calls[0]["success"])
        self.assertEqual(report.critical_error["error_type"], "parameter_error")

    def test_sqlite_persistence_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "agentshield.db")
            report = self.agent.execute("Calculate 12 * 8")
            execution_id = store.save_execution(report)
            store.save_execution_details(execution_id, "")
            store.save_memory(execution_id, "answer", report.answer)
            execution = store.get_execution(execution_id)
            self.assertIsNotNone(execution)
            self.assertEqual(execution["query"], "Calculate 12 * 8")
            self.assertEqual(execution["answer"], "calculator: 96")
            self.assertEqual(len(store.list_executions()), 1)
            self.assertEqual(store.list_memory()[0]["key"], "answer")

    def test_web_search_intent_selects_search_tool(self):
        use_tool, tools = ToolSelector().decide("Search the web for the latest Python release")
        self.assertTrue(use_tool)
        self.assertEqual(tools, ["web_search"])

    def test_web_search_query_is_extracted(self):
        report = self.agent.execute("Search the web for Python 3.15 release")
        self.assertTrue(report.use_tool)
        self.assertEqual(report.selected_tools, ["web_search"])
        self.assertEqual(
            report.tool_calls[0]["arguments"]["query"],
            "Python 3.15 release",
        )

    def test_missing_web_search_query_is_diagnosed(self):
        report = self.agent.execute("Search the web")
        self.assertTrue(report.use_tool)
        self.assertEqual(report.selected_tools, ["web_search"])
        self.assertFalse(report.tool_calls[0]["success"])
        self.assertEqual(report.critical_error["error_type"], "parameter_error")

    def test_hybrid_router_exposes_confidence(self):
        confidence = ToolSelector().confidence("Search the web for quantum computing")
        self.assertGreater(confidence["web_search"], 0.8)
        self.assertGreater(confidence["web_search"], confidence["calculator"])

    def test_hybrid_router_handles_ambiguous_no_tool_request(self):
        use_tool, tools = ToolSelector().decide("Explain the difference between APIs")
        self.assertFalse(use_tool)
        self.assertEqual(tools, [])


if __name__ == "__main__":
    unittest.main()
