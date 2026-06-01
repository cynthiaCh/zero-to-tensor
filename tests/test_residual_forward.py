import unittest

import numpy as np

from experiments.residual_forward import relu, residual_block_forward


class ResidualForwardTest(unittest.TestCase):
    def test_residual_block_adds_input_back_to_branch_output(self):
        x = np.array([[1.0, -2.0, 3.0]])
        w1 = np.eye(3)
        b1 = np.zeros(3)
        w2 = np.eye(3)
        b2 = np.zeros(3)

        y, branch = residual_block_forward(x, w1, b1, w2, b2)

        expected_branch = relu(x)
        np.testing.assert_allclose(branch, expected_branch)
        np.testing.assert_allclose(y, expected_branch + x)
        self.assertEqual(y.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
