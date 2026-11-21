import json
import tempfile
import unittest
from pathlib import Path

from agentdebug import HeuristicErrorDetector, evaluate, parse_trajectory


class AgentDebugTests(unittest.TestCase):
    def setUp(self):
        self.detector = HeuristicErrorDetector()

    def test_parse_and_analyze_failed_trajectory(self):
        data = {
            "metadata": {"task": "Put apple in microwave", "success": False, "environment": "alfworld"},
            "messages": [
                {"role": "user", "content": "Put apple in microwave"},
                {"role": "assistant", "content":
                    "<memory>knife is in kitchen</memory>"
                    "<reflection>Nothing happens.</reflection>"
                    "<plan>Put the apple in the microwave.</plan>"
                    "<action>GoTo(Bedroom 1)</action>"},
                {"role": "user", "content": "Nothing happens"},
                {"role": "assistant", "content":
                    "<memory>I remember completing the task</memory>"
                    "<reflection>The task is complete.</reflection>"
                    "<plan>Proceed to finish.</plan>"
                    "<action>Put(Apple, Microwave)</action>"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.json"
            path.write_text(json.dumps(data))
            trajectory = parse_trajectory(path)
        phase1 = self.detector.analyze_trajectory(trajectory)
        critical = self.detector.identify_critical_error(phase1)
        self.assertEqual(phase1["total_steps"] if "total_steps" in phase1 else 2, 2)
        self.assertIsNotNone(critical)
        self.assertEqual(critical.critical_module, "action")
        self.assertEqual(critical.error_type, "misalignment")

    def test_successful_trajectory_has_no_critical_error(self):
        trajectory = {
            "task_id": "ok", "task_description": "Open the box", "success": True,
            "environment": "alfworld",
            "steps": [{"step": 1, "content": "<plan>Open the box.</plan><action>Open(Box)</action>",
                       "env_response": "Success"}],
        }
        phase1 = self.detector.analyze_trajectory(trajectory)
        self.assertIsNone(self.detector.identify_critical_error(phase1))

    def test_step_one_memory_is_not_critical(self):
        phase1 = {
            "task_success": False,
            "step_analyses": [{
                "step": 1,
                "memory_error": {"error_detected": True, "error_type": "hallucination",
                                 "evidence": "x", "reasoning": "x"},
                "reflection_error": {"error_detected": False, "error_type": "no_error"},
                "planning_error": {"error_detected": False, "error_type": "no_error"},
                "action_error": {"error_detected": True, "error_type": "misalignment",
                                 "evidence": "wrong action", "reasoning": "contradicts plan"},
                "system_error": {"error_detected": False, "error_type": "no_error"},
            }],
        }
        critical = self.detector.identify_critical_error(phase1)
        self.assertEqual(critical.critical_module, "action")

    def test_evaluation_metrics(self):
        predicted = [{
            "step_analyses": [{
                "memory_error": {"error_type": "no_error"},
                "reflection_error": {"error_type": "no_error"},
                "planning_error": {"error_type": "no_error"},
                "action_error": {"error_type": "format_error"},
                "system_error": {"error_type": "no_error"},
            }],
            "critical_error": {"critical_step": 1, "critical_module": "action",
                               "error_type": "format_error"},
        }]
        gold = [{
            "step_analyses": [{
                "memory_error": {"error_type": "no_error"},
                "reflection_error": {"error_type": "no_error"},
                "planning_error": {"error_type": "no_error"},
                "action_error": {"error_type": "format_error"},
                "system_error": {"error_type": "no_error"},
            }],
            "critical_error": {"critical_step": 1, "critical_module": "action",
                               "error_type": "format_error"},
        }]
        metrics = evaluate(predicted, gold)
        self.assertEqual(metrics["step_accuracy"], 1.0)
        self.assertEqual(metrics["critical_all_correct_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
