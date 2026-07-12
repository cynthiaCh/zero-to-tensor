import unittest

import numpy as np

from experiments.rnn_step_by_step import (
    make_demo_params,
    make_demo_sequence,
    rnn_forward,
    rnn_step,
)


class RnnStepByStepTest(unittest.TestCase):
    def test_single_step_uses_current_input_and_previous_hidden(self):
        params = make_demo_params()
        x_t = np.array([[2.0]])
        h_prev = np.array([[0.5]])

        h_t, y_t, hidden_raw = rnn_step(x_t, h_prev, params)

        np.testing.assert_allclose(hidden_raw, np.array([[1.05]]))
        np.testing.assert_allclose(h_t, np.tanh(np.array([[1.05]])))
        np.testing.assert_allclose(y_t, h_t * 2.0)

    def test_forward_runs_in_time_order_and_carries_hidden_state(self):
        sequence = make_demo_sequence()
        params = make_demo_params()

        steps = rnn_forward(sequence, params)

        h0 = np.array([[0.0]])
        raw0 = np.array([[0.5]])
        h1 = np.tanh(raw0)
        raw1 = np.array([[1.0]]) + h1 * 0.1
        h2 = np.tanh(raw1)
        raw2 = np.array([[1.5]]) + h2 * 0.1
        h3 = np.tanh(raw2)

        self.assertEqual([step["time_index"] for step in steps], [0, 1, 2])
        np.testing.assert_allclose(steps[0]["h_prev"], h0)
        np.testing.assert_allclose(steps[0]["hidden_raw"], raw0)
        np.testing.assert_allclose(steps[0]["h_t"], h1)
        np.testing.assert_allclose(steps[1]["h_prev"], h1)
        np.testing.assert_allclose(steps[1]["hidden_raw"], raw1)
        np.testing.assert_allclose(steps[1]["h_t"], h2)
        np.testing.assert_allclose(steps[2]["h_prev"], h2)
        np.testing.assert_allclose(steps[2]["hidden_raw"], raw2)
        np.testing.assert_allclose(steps[2]["h_t"], h3)


if __name__ == "__main__":
    unittest.main()
