"""Lightweight token/state regression tests for qwen38.chat.

These tests intentionally use no model weights and no Metal execution.
"""

from __future__ import annotations

import unittest

from qwen38.chat import (
    HistoryTurn,
    TokenLedger,
    exact_suffix,
    reconcile_canonical_history,
    response_replay_detected,
)


class TokenLedgerTests(unittest.TestCase):
    def test_later_turn_suffix_does_not_add_bos(self) -> None:
        bos = 101
        self.assertEqual(exact_suffix([bos, 10, 20, 30], [bos, 10]), [20, 30])
        self.assertNotIn(bos, exact_suffix([bos, 10, 20, 30], [bos, 10]))

    def test_cached_logical_prefix_accounting(self) -> None:
        ledger = TokenLedger()
        ledger.commit([101, 10, 20], [30, 31])
        self.assertEqual(ledger.cached_token_ids, [101, 10, 20, 30, 31])
        self.assertEqual(
            ledger.suffix_for([101, 10, 20, 30, 31, 40, 41]),
            [40, 41],
        )

    def test_stale_prefix_is_rejected(self) -> None:
        ledger = TokenLedger()
        ledger.commit([1, 2, 3], [4])
        with self.assertRaisesRegex(RuntimeError, "exact prefix"):
            ledger.suffix_for([1, 2, 9, 4, 5])

    def test_reset_clears_transcript_token_state(self) -> None:
        ledger = TokenLedger()
        ledger.commit([1, 2], [3, 4])
        ledger.reset()
        self.assertEqual(ledger.cached_token_ids, [])
        self.assertEqual(ledger.suffix_for([9, 10]), [9, 10])

    def test_replay_regression_is_detected_at_token_level(self) -> None:
        self.assertTrue(response_replay_detected(list(range(8)), list(range(9))))
        self.assertFalse(response_replay_detected([8, 7], [7, 8, 9]))
        self.assertFalse(response_replay_detected([], [7, 8, 9]))

    def test_authoritative_ids_repair_rendered_assistant_content(self) -> None:
        history = [
            HistoryTurn(
                user_text="old",
                prompt_token_count=2,
                generated_ids=[20, 21, 22],
                assistant_text="replacement text",
            )
        ]
        # The rendered template lost token 21 and re-encoded token 22 as 23;
        # the close boundary is still token 99.
        canonical = {"token_ids": [1, 2, 20, 23, 99, 30]}
        adjusted = reconcile_canonical_history(canonical, history, {99})
        self.assertEqual(adjusted["token_ids"], [1, 2, 20, 21, 22, 99, 30])
        self.assertTrue(adjusted["cache_boundary_adjustments"])

    def test_authoritative_ids_keep_one_eos_boundary(self) -> None:
        history = [
            HistoryTurn(
                user_text="old",
                prompt_token_count=2,
                generated_ids=[20, 99],
                assistant_text="answer",
            )
        ]
        canonical = {"token_ids": [1, 2, 20, 99, 99, 30]}
        adjusted = reconcile_canonical_history(canonical, history, {99})
        self.assertEqual(adjusted["token_ids"], [1, 2, 20, 99, 30])

    def test_later_turns_use_authoritative_positions(self) -> None:
        # Turn 1 renders shorter than it was generated; turn 2's recorded
        # prompt boundary is a cache position and must not be shifted again.
        history = [
            HistoryTurn("old", 2, [20, 21, 22, 99], "a"),
            HistoryTurn("old", 7, [41, 99], "b"),
        ]
        canonical = {"token_ids": [1, 2, 20, 23, 99, 30, 40, 50, 99, 60]}
        adjusted = reconcile_canonical_history(canonical, history, {99})
        self.assertEqual(adjusted["token_ids"], [1, 2, 20, 21, 22, 99, 30, 41, 99, 60])

    def test_later_turns_after_longer_render(self) -> None:
        history = [
            HistoryTurn("old", 2, [20, 99], "a"),
            HistoryTurn("old", 5, [41, 42, 99], "b"),
        ]
        # Turn 1 rendered as three tokens instead of one.
        canonical = {"token_ids": [1, 2, 20, 23, 24, 99, 30, 40, 99, 60]}
        adjusted = reconcile_canonical_history(canonical, history, {99})
        self.assertEqual(adjusted["token_ids"], [1, 2, 20, 99, 30, 41, 42, 99, 60])


if __name__ == "__main__":
    unittest.main()
