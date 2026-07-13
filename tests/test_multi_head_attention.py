import unittest

import numpy as np

from experiments.multi_head_attention import (
    make_demo_inputs,
    make_demo_params,
    multi_head_attention,
    single_head_attention,
)


class MultiHeadAttentionTest(unittest.TestCase):
    def test_each_head_has_its_own_qkv_projections(self):
        inputs = make_demo_inputs()
        params = make_demo_params()

        first_head = single_head_attention(inputs, params["heads"][0])
        second_head = single_head_attention(inputs, params["heads"][1])

        np.testing.assert_allclose(first_head[0], np.array([[1.0], [0.0], [1.0]]))
        np.testing.assert_allclose(second_head[0], np.array([[0.0], [1.0], [1.0]]))
        self.assertFalse(np.allclose(first_head[-1], second_head[-1]))

    def test_outputs_are_concatenated_then_projected(self):
        inputs = make_demo_inputs()
        params = make_demo_params()

        head_results, concatenated, outputs = multi_head_attention(inputs, params)
        expected_concatenated = np.concatenate([result[-1] for result in head_results], axis=1)

        self.assertEqual(concatenated.shape, (3, 2))
        np.testing.assert_allclose(concatenated, expected_concatenated)
        np.testing.assert_allclose(outputs, concatenated @ params["wo"])


if __name__ == "__main__":
    unittest.main()
