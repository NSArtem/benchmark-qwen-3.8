import unittest

from report import format_report


class ReportTests(unittest.TestCase):
    def test_columns_and_title(self):
        text = format_report([{"name": "a", "amount": 1.5}, {"name": "bb", "qty": 2}], title="Sales")
        self.assertEqual(
            text.splitlines(),
            ["Sales", "=====", "name  amount  qty", "a     1.50", "bb            2"],
        )

    def test_without_title(self):
        self.assertEqual(format_report([{"x": 1}]).splitlines(), ["x", "1"])


if __name__ == "__main__":
    unittest.main()
