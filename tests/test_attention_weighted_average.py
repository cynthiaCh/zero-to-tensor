import unittest

import numpy as np

from experiments.attention_weighted_average import (
    make_demo_inputs,
    scaled_dot_product_attention,
    softmax,
)


class AttentionWeightedAverageTest(unittest.TestCase):
    def test_softmax_weights_sum_to_one(self):
        weights = softmax(np.array([[2.0, 1.0, 0.0]]))

        np.testing.assert_allclose(weights.sum(axis=1), np.array([1.0]))
        self.assertGreater(weights[0, 0], weights[0, 1])
        self.assertGreater(weights[0, 1], weights[0, 2])

    def test_attention_returns_weighted_average_of_values(self):
        query, keys, values = make_demo_inputs()

        scores, weights, context = scaled_dot_product_attention(query, keys, values)
        expected_scores = np.array([[2.0, 1.0, 0.0]]) / np.sqrt(2.0)
        expected_weights = softmax(expected_scores)

        np.testing.assert_allclose(scores, expected_scores)
        np.testing.assert_allclose(weights, expected_weights)
        np.testing.assert_allclose(context, expected_weights @ values)


if __name__ == "__main__":
    unittest.main()
