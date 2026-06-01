import unittest

import numpy as np

from experiments.train_residual_linear import (
    forward,
    loss_and_grads,
    make_dataset,
    train,
)


class TrainResidualLinearTest(unittest.TestCase):
    def test_forward_keeps_input_and_adds_learned_branch(self):
        x = np.array([[2.0], [3.0]])
        params = {"w": np.array([[0.5]]), "b": np.array([[1.0]])}

        y_pred, branch = forward(x, params)

        np.testing.assert_allclose(branch, np.array([[2.0], [2.5]]))
        np.testing.assert_allclose(y_pred, x + branch)

    def test_training_reduces_loss_and_learns_to_double_x(self):
        x, y = make_dataset()
        params, history = train(x, y, steps=200, learning_rate=0.05)

        self.assertLess(history[-1], history[0])
        self.assertLess(history[-1], 1e-3)
        self.assertAlmostEqual(params["w"].item(), 1.0, places=2)
        self.assertAlmostEqual(params["b"].item(), 0.0, places=2)

    def test_loss_gradient_points_to_smaller_error(self):
        x, y = make_dataset()
        params = {"w": np.array([[0.0]]), "b": np.array([[0.0]])}
        loss, grads = loss_and_grads(x, y, params)

        next_params = {
            "w": params["w"] - 0.05 * grads["w"],
            "b": params["b"] - 0.05 * grads["b"],
        }
        next_loss, _ = loss_and_grads(x, y, next_params)

        self.assertLess(next_loss, loss)


if __name__ == "__main__":
    unittest.main()
