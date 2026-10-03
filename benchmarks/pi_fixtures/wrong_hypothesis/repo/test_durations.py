import unittest

from durations import format_duration, parse_duration


class DurationTests(unittest.TestCase):
    def test_parse_rejects_garbage(self):
        for text in ("", "10", "1x", "h1"):
            with self.assertRaises(ValueError):
                parse_duration(text)

    def test_round_trip(self):
        for text in ("45s", "1h30m", "2d", "3h", "1d2h3m4s"):
            self.assertEqual(format_duration(parse_duration(text)), text)

    def test_zero(self):
        self.assertEqual(format_duration(0), "0s")


if __name__ == "__main__":
    unittest.main()
