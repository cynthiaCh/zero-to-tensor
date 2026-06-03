import unittest

import numpy as np

from experiments.two_layer_linear_forward import (
    make_demo_input,
    make_demo_params,
    plain_forward,
    residual_forward,
    two_layer_branch,
)


class TwoLayerLinearForwardTest(unittest.TestCase):
    def test_two_layer_branch_passes_data_through_hidden_layer(self):
        x = make_demo_input()
        params = make_demo_params()

        branch, hidden = two_layer_branch(x, params)

        np.testing.assert_allclose(hidden, np.array([[2.0], [4.0], [6.0]]))
        np.testing.assert_allclose(branch, np.array([[6.0], [12.0], [18.0]]))
        self.assertEqual(hidden.shape, (3, 1))
        self.assertEqual(branch.shape, (3, 1))

    def test_plain_forward_outputs_only_branch(self):
        x = make_demo_input()
        params = make_demo_params()

        y_pred, branch, hidden = plain_forward(x, params)

        np.testing.assert_allclose(y_pred, branch)
        self.assertEqual(y_pred.shape, x.shape)
        self.assertEqual(hidden.shape, x.shape)

    def test_residual_forward_adds_input_back(self):
        x = make_demo_input()
        params = make_demo_params()

        y_pred, branch, hidden = residual_forward(x, params)

        np.testing.assert_allclose(y_pred, x + branch)
        np.testing.assert_allclose(y_pred, np.array([[7.0], [14.0], [21.0]]))
        self.assertEqual(y_pred.shape, x.shape)
        self.assertEqual(hidden.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
