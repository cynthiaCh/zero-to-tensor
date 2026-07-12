import unittest

import numpy as np

from experiments.autoregressive_generation import (
    generate,
    make_transition_logits,
    predict_next_token,
)


class AutoregressiveGenerationTest(unittest.TestCase):
    def test_prediction_uses_the_last_token_in_the_prefix(self):
        transition_logits = make_transition_logits()

        next_token, probabilities = predict_next_token(["<BOS>", "我"], transition_logits)

        self.assertEqual(next_token, "爱")
        np.testing.assert_allclose(probabilities.sum(), 1.0)

    def test_generation_appends_each_prediction_until_eos(self):
        transition_logits = make_transition_logits()

        generated_tokens, steps = generate(transition_logits)

        self.assertEqual(generated_tokens, ["<BOS>", "我", "爱", "学习", "<EOS>"])
        self.assertEqual([step["next_token"] for step in steps], ["我", "爱", "学习", "<EOS>"])
        self.assertEqual(steps[2]["prefix"], ["<BOS>", "我", "爱"])


if __name__ == "__main__":
    unittest.main()
