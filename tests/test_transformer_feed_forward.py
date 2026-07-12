import unittest

import numpy as np

from experiments.transformer_feed_forward import (
    feed_forward,
    make_demo_inputs,
    make_demo_params,
    relu,
)


class TransformerFeedForwardTest(unittest.TestCase):
    def test_relu_replaces_negative_values_with_zero(self):
        result = relu(np.array([[-1.0, 0.0, 2.0]]))

        np.testing.assert_allclose(result, np.array([[0.0, 0.0, 2.0]]))

    def test_feed_forward_uses_the_same_network_for_each_position(self):
        inputs = make_demo_inputs()
        params = make_demo_params()

        hidden_raw, hidden, outputs = feed_forward(inputs, params)

        np.testing.assert_allclose(hidden_raw[0], np.array([2.0, 0.5, -0.5]))
        np.testing.assert_allclose(hidden[0], np.array([2.0, 0.5, 0.0]))
        np.testing.assert_allclose(outputs[0], np.array([1.6, 1.3]))

        _, single_hidden, single_output = feed_forward(inputs[1:2], params)
        np.testing.assert_allclose(hidden[1], single_hidden[0])
        np.testing.assert_allclose(outputs[1], single_output[0])


if __name__ == "__main__":
    unittest.main()
