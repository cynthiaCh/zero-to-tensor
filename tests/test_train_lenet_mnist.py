import unittest

import torch

from experiments.train_lenet_mnist import LeNet


class TrainLeNetMnistTest(unittest.TestCase):
    def test_lenet_outputs_one_score_per_digit(self):
        model = LeNet()
        images = torch.zeros(4, 1, 28, 28)

        scores = model(images)

        self.assertEqual(scores.shape, (4, 10))


if __name__ == "__main__":
    unittest.main()
