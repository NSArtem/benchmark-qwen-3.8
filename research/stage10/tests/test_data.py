from __future__ import annotations

import unittest

from stage10.data import Document, make_domain_arrays


class TinyTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool):
        if add_special_tokens:
            raise AssertionError("test requires no special tokens")
        return {"input_ids": [int(token) for token in text.split()]}


class DataWindowTests(unittest.TestCase):
    def test_fake_corpus_is_joined_and_cut_into_consecutive_windows(self):
        tokenizer = TinyTokenizer()
        eval_docs = [Document("eval-a", "0 1 2 3 4"), Document("eval-b", "5 6 7 8")]
        calibration_docs = [Document("cal-a", "10 11 12 13 14 15 16 17")]
        eval_array, calibration_array = make_domain_arrays(
            eval_docs,
            calibration_docs,
            tokenizer,
            sequence_length=4,
            eval_count=2,
            calibration_count=2,
        )
        self.assertEqual(eval_array.tolist(), [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(calibration_array.tolist(), [[10, 11, 12, 13], [14, 15, 16, 17]])

    def test_eval_and_calibration_cannot_share_a_source_document(self):
        shared_eval = [Document("same-doc", "0 1 2 3")]
        shared_calibration = [Document("same-doc", "4 5 6 7")]
        with self.assertRaisesRegex(ValueError, "share source documents"):
            make_domain_arrays(
                shared_eval,
                shared_calibration,
                TinyTokenizer(),
                sequence_length=4,
                eval_count=1,
                calibration_count=1,
            )


if __name__ == "__main__":
    unittest.main()
