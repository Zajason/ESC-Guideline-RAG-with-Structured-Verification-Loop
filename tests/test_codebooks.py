import unittest

from esc_rag.codebooks import QUERY_A_ALLOWED, QUERY_B_ALLOWED, binary, clamp_int, invalid_codes


class CodebookTests(unittest.TestCase):
    def test_binary_coerces_positive_to_one(self):
        self.assertEqual(binary(2), 1)
        self.assertEqual(binary("1"), 1)
        self.assertEqual(binary(0), 0)
        self.assertEqual(binary(None), 0)

    def test_ctpa_query_a_is_binary(self):
        bad = invalid_codes({"er_ctpa": 2}, QUERY_A_ALLOWED)
        self.assertEqual(bad, {"er_ctpa": 2})

    def test_query_b_coro_is_binary(self):
        bad = invalid_codes({"coro": 2}, QUERY_B_ALLOWED)
        self.assertEqual(bad, {"coro": 2})

    def test_clamp_int_uses_default_for_invalid_value(self):
        self.assertEqual(clamp_int(8, {0, 1}, default=0), 0)


if __name__ == "__main__":
    unittest.main()
