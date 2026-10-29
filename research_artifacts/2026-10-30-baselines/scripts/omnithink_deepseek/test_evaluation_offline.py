import importlib.util
import json
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('omni_eval',Path(__file__).with_name('evaluate_reference.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class Checks(unittest.TestCase):
    def test_free_precision_and_differences_beyond_one(self):
        self.assertEqual(module.parse_delta('{"a": -2.735}',{'a':8.2}),{'a':-2.735})
        self.assertEqual(module.parse_delta('{"a": 1.8}',{'a':8.2}),{'a':1.8})
    def test_implied_scores_must_fit_scale(self):
        for text in ['{"a":2}','{"a":-9}','{"a":true}','{"a":NaN}','{"a":Infinity}']:
            with self.assertRaises(ValueError):module.parse_delta(text,{'a':8.2})
    def test_duplicate_missing_or_extra_metric_rejected(self):
        for text in ['{"a":0,"a":1}','{}','{"a":0,"b":0}']:
            with self.assertRaises(ValueError):module.parse_delta(text,{'a':8.2})
    def test_same_published_prompt_hash_is_preserved(self):
        for suite in module.METRICS:
            prompt=(module.PROTOCOLS/f'{suite}_ablation_prompt.txt').read_text(encoding='utf-8')
            self.assertIn('Do not replace them with a universal constant',prompt)
            for metric in module.METRICS[suite]:self.assertIn(metric,prompt)

if __name__=='__main__':unittest.main(verbosity=2)
