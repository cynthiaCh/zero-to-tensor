import unittest

import numpy as np

from experiments.qkv_self_attention import (
    make_demo_inputs,
    make_demo_params,
    project_qkv,
    self_attention,
)


class QkvSelfAttentionTest(unittest.TestCase):
    def test_qkv_are_different_projections_of_the_same_input(self):
        inputs = make_demo_inputs()
        params = make_demo_params()

        queries, keys, values = project_qkv(inputs, params)

        np.testing.assert_allclose(queries, np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        np.testing.assert_allclose(keys, np.array([[1.0, 1.0], [0.0, 1.0], [1.0, 2.0]]))
        np.testing.assert_allclose(values, np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 1.0]]))

    def test_attention_uses_qk_for_weights_and_v_for_output(self):
        inputs = make_demo_inputs()
        params = make_demo_params()

        queries, keys, values, scores, weights, outputs = self_attention(inputs, params)

        np.testing.assert_allclose(scores, queries @ keys.T / np.sqrt(2.0))
        np.testing.assert_allclose(weights.sum(axis=1), np.ones(3))
        np.testing.assert_allclose(outputs, weights @ values)


if __name__ == "__main__":
    unittest.main()
