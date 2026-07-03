import unittest

from esc_rag.metrics import per_class_recall, prognosis_metrics


class MetricsTests(unittest.TestCase):
    def test_prognosis_metrics_reward_near_misses(self):
        pairs = [(4, 4), (5, 4), (8, 8), (6, 8)]
        metrics = prognosis_metrics(pairs)
        self.assertEqual(metrics["n"], 4)
        self.assertEqual(metrics["exact_accuracy"], 0.5)
        self.assertEqual(metrics["within_1"], 0.75)
        self.assertEqual(metrics["mae"], 0.75)

    def test_per_class_recall(self):
        pairs = [(8, 8), (6, 8), (7, 7), (5, 4)]
        recall = per_class_recall(pairs)
        self.assertEqual(recall[8], 0.5)
        self.assertEqual(recall[7], 1.0)
        self.assertEqual(recall[4], 0.0)


if __name__ == "__main__":
    unittest.main()
