import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image

from experiments.train_lenet_mnist import LeNet, save_model, visualize_kernels


class TrainLeNetMnistTest(unittest.TestCase):
    def test_lenet_outputs_one_score_per_digit(self):
        model = LeNet()
        images = torch.zeros(4, 1, 28, 28)

        scores = model(images)

        self.assertEqual(scores.shape, (4, 10))

    def test_save_model_writes_loadable_state_dict(self):
        model = LeNet()

        with TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "nested" / "lenet.pth"

            save_model(model, model_path)

            saved_state = torch.load(model_path, map_location="cpu")

        self.assertEqual(saved_state.keys(), model.state_dict().keys())

    def test_visualize_kernels_writes_conv_kernel_images(self):
        model = LeNet()

        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "epoch_00_init"

            visualize_kernels(model, output_dir)

            conv1_path = output_dir / "conv1.png"
            conv2_path = output_dir / "conv2.png"

            with Image.open(conv1_path) as conv1_image:
                self.assertEqual(conv1_image.format, "PNG")
            with Image.open(conv2_path) as conv2_image:
                self.assertEqual(conv2_image.format, "PNG")


if __name__ == "__main__":
    unittest.main()
