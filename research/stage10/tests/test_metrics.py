import unittest

import numpy as np

from stage10.evaluate import metrics
from stage10.gen_check import json_valid, ngram_stats, score
from stage10.make_table import render


class MetricsTest(unittest.TestCase):
    def test_identical_candidate(self):
        rng = np.random.default_rng(0)
        nll = rng.random((3, 5)).astype(np.float32)
        top1 = rng.integers(0, 9, (3, 5))
        logp = np.log(rng.dirichlet(np.ones(32), (3, 5))).astype(np.float32)
        m = metrics(nll, top1, logp, nll, top1, logp)
        self.assertEqual(m["delta_nll"], 0.0)
        self.assertEqual(m["top1_agree"], 1.0)
        self.assertAlmostEqual(m["kl_top32"], 0.0, places=7)
        self.assertAlmostEqual(m["ppl"], float(np.exp(np.mean(nll, dtype=np.float64))))

    def test_kl_top32_positive_when_candidate_differs(self):
        ref = np.log(np.full((1, 1, 32), 1 / 32))
        cand = ref - 0.1
        m = metrics(np.ones((1, 1)), np.zeros((1, 1)), cand, np.ones((1, 1)), np.zeros((1, 1)), ref)
        self.assertAlmostEqual(m["kl_top32"], 0.1, places=6)


class GenCheckTest(unittest.TestCase):
    def test_ngram_stats(self):
        self.assertEqual(ngram_stats([1, 2, 3, 4, 1, 2, 3, 4]), (4 / 5, 2))
        self.assertEqual(ngram_stats([1, 2]), (0.0, 0))

    def test_json_valid(self):
        self.assertTrue(json_valid('```json\n{"a": [1]}\n```'))
        self.assertTrue(json_valid(' [1, 2] '))
        self.assertFalse(json_valid("Here: {\"a\": 1}"))

    def test_prompt_specific_fields(self):
        self.assertTrue(score("P2", "It moves 187 minutes", [1, 2, 3, 4], True)["p2_has_187"])
        self.assertNotIn("p2_has_187", score("P1", "x", [], False))


class TableTest(unittest.TestCase):
    def test_render_handles_missing_sections(self):
        text = render([{"exp_id": "E00", "status": "OK", "recipe": "bf16-ref"}])
        self.assertIn("| E00 | OK | bf16-ref |", text)


if __name__ == "__main__":
    unittest.main()
