import unittest

import numpy as np

from experiments.compare_plain_and_residual_linear import (
    make_dataset,
    plain_forward,
    residual_forward,
    train,
)


class ComparePlainAndResidualLinearTest(unittest.TestCase):
    def test_plain_forward_uses_only_learned_branch(self):
        x = np.array([[2.0], [3.0]])
        params = {"w": np.array([[0.5]]), "b": np.array([[1.0]])}

        y_pred, branch = plain_forward(x, params)

        np.testing.assert_allclose(branch, np.array([[2.0], [2.5]]))
        np.testing.assert_allclose(y_pred, branch)

    def test_residual_forward_adds_input_to_learned_branch(self):
        x = np.array([[2.0], [3.0]])
        params = {"w": np.array([[0.5]]), "b": np.array([[1.0]])}

        y_pred, branch = residual_forward(x, params)

        np.testing.assert_allclose(branch, np.array([[2.0], [2.5]]))
        np.testing.assert_allclose(y_pred, x + branch)

    def test_plain_model_learns_full_answer_weight(self):
        x, y = make_dataset()
        params, history = train(plain_forward, x, y, steps=200, learning_rate=0.05)

        self.assertLess(history[-1], 1e-3)
        self.assertAlmostEqual(params["w"].item(), 2.0, places=2)
        self.assertAlmostEqual(params["b"].item(), 0.0, places=2)

    def test_residual_model_learns_correction_weight(self):
        x, y = make_dataset()
        params, history = train(residual_forward, x, y, steps=200, learning_rate=0.05)

        self.assertLess(history[-1], 1e-3)
        self.assertAlmostEqual(params["w"].item(), 1.0, places=2)
        self.assertAlmostEqual(params["b"].item(), 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
