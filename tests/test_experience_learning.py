"""Unit tests for Thought Pattern Distillation and pattern-store refresh.

Run: python -m unittest tests.test_experience_learning
All LLM calls are mocked; no API key or network is required.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.pattern_matching.pattern_matcher import PatternMatcher
from core.thought_distillation.pattern_distiller import PatternDistiller


VALID_PATTERN = {
    "task_description": "A user requests a product for a specific use case.",
    "solution_description": "Translate the use case into product attributes before retrieval.",
    "thought_template": "1. Identify use case. 2. Find attributes. 3. Retrieve products."
}


class ExperienceLearningTests(unittest.TestCase):
    def test_success_creates_pattern(self):
        with tempfile.TemporaryDirectory() as tmp:
            distiller = PatternDistiller(tmp)
            with patch("core.thought_distillation.pattern_distiller.get_completion",
                       return_value=json.dumps(VALID_PATTERN)):
                key = distiller.distill_from_success(
                    {"query": "jacket for gallery", "execution_log": []},
                    "jacket for gallery"
                )
            self.assertEqual(key, "template_0")
            saved = json.loads((Path(tmp) / "patterns.json").read_text(encoding="utf-8"))
            self.assertIn(key, saved)

    def test_success_can_update_selected_similar_pattern(self):
        with tempfile.TemporaryDirectory() as tmp:
            distiller = PatternDistiller(tmp)
            distiller.patterns = {"template_0": dict(VALID_PATTERN)}
            response = dict(VALID_PATTERN, action="update", solution_description="Improved approach.")
            with patch("core.thought_distillation.pattern_distiller.get_completion",
                       return_value=json.dumps(response)):
                key = distiller.distill_from_success(
                    {"query": "another gallery jacket", "execution_log": []},
                    "another gallery jacket",
                    "template_0"
                )
            self.assertEqual(key, "template_0")
            self.assertEqual(len(distiller.patterns), 1)
            self.assertEqual(distiller.patterns[key]["solution_description"], "Improved approach.")

    def test_expert_correction_creates_pattern_when_none_was_matched(self):
        with tempfile.TemporaryDirectory() as tmp:
            distiller = PatternDistiller(tmp)
            with patch("core.thought_distillation.pattern_distiller.get_completion",
                       return_value=json.dumps(VALID_PATTERN)):
                key = distiller.distill_from_failure(
                    {"query": "unseen task", "execution_log": []},
                    "The plan skipped attribute discovery.",
                    None
                )
            self.assertEqual(key, "template_0")
            self.assertIn(key, distiller.patterns)

    def test_expert_correction_refines_existing_pattern(self):
        with tempfile.TemporaryDirectory() as tmp:
            distiller = PatternDistiller(tmp)
            distiller.patterns = {"template_4": dict(VALID_PATTERN)}
            response = dict(VALID_PATTERN, thought_template="1. Clarify. 2. Search. 3. Retrieve. 4. Verify.")
            with patch("core.thought_distillation.pattern_distiller.get_completion",
                       return_value=json.dumps(response)):
                key = distiller.distill_from_failure(
                    {"query": "failed task", "execution_log": []},
                    "The old plan omitted verification.",
                    "template_4"
                )
            self.assertEqual(key, "template_4")
            self.assertIn("4. Verify.", distiller.patterns[key]["thought_template"])

    def test_matcher_reloads_patterns_between_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "patterns.json"
            path.write_text(json.dumps({"template_0": dict(VALID_PATTERN)}), encoding="utf-8")
            matcher = PatternMatcher(str(path))
            path.write_text(json.dumps({
                "template_0": dict(VALID_PATTERN),
                "template_1": dict(VALID_PATTERN, task_description="A different task type.")
            }), encoding="utf-8")
            with patch("core.pattern_matching.pattern_matcher.get_completion",
                       return_value="Reasoning... {1}"):
                key, pattern = matcher.select_best_pattern("different task")
            self.assertEqual(key, "template_1")
            self.assertEqual(pattern["task_description"], "A different task type.")


if __name__ == "__main__":
    unittest.main()
