import unittest

from gta import (
    answer_correct,
    argument_accuracy,
    f1_for_category,
    ground_truth_steps,
    load_gta_dataset,
    tool_accuracy,
)


class GTATests(unittest.TestCase):
    def setUp(self):
        self.dataset = load_gta_dataset("official_gta_dataset/dataset.json")

    def test_official_dataset_size(self):
        self.assertEqual(len(self.dataset), 229)
        self.assertEqual(sum(len(ground_truth_steps(task)) for task in self.dataset.values()), 557)

    def test_step_metrics(self):
        gold = [{"name": "OCR", "arguments": {"image": "image/a.jpg"}}]
        exact = [{"name": "OCR", "arguments": {"image": "image/a.jpg"}}]
        wrong = [{"name": "Plot", "arguments": {"image": "image/a.jpg"}}]
        self.assertEqual(tool_accuracy(exact, gold), 1.0)
        self.assertEqual(argument_accuracy(exact, gold), 1.0)
        self.assertEqual(tool_accuracy(wrong, gold), 0.0)

    def test_answer_whitelist_blacklist(self):
        gt = {"whitelist": [["Paris"]], "blacklist": ["London"]}
        self.assertTrue(answer_correct("The city is Paris.", gt))
        self.assertFalse(answer_correct("Paris or London.", gt))
        self.assertFalse(answer_correct("Berlin.", gt))

    def test_category_f1(self):
        self.assertEqual(f1_for_category({"OCR", "Plot"}, {"OCR", "Calculator"}, "P"), 1.0)
        self.assertEqual(f1_for_category({"OCR"}, set(), "P"), 0.0)
        self.assertEqual(f1_for_category(set(), {"OCR"}, "P"), 0.0)


if __name__ == "__main__":
    unittest.main()
