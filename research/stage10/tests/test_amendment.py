import unittest
from unittest.mock import patch

from stage10.build import EXPERIMENTS, RECIPES, module_records
from stage10.gen_check import MAX_TOKENS
from stage10.ladder import contexts_for


class AmendmentTest(unittest.TestCase):
    def test_generation_limit(self):
        self.assertEqual(MAX_TOKENS, 1024)

    def test_all_ladder_lengths_for_all_sizes(self):
        for size in (8.0, 9.0, 10.18, 11.0):
            self.assertEqual(contexts_for(size), [4096, 8192, 16384, 24576, 32768])

    def test_recipe_dtype_written_to_module_config(self):
        with patch('stage10.build.quantized_paths', return_value=['linear']):
            for exp, name in EXPERIMENTS.items():
                expected = 'bfloat16' if 'E07' <= exp <= 'E12' else 'float32'
                self.assertEqual(module_records(None, RECIPES[name])[0]['rotate_dtype'], expected)


if __name__ == '__main__':
    unittest.main()
