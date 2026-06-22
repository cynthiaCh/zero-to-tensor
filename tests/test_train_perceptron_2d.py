import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from experiments.train_perceptron_2d import (
    make_training_points,
    predict_label,
    score_point,
    train_perceptron,
)


class TrainPerceptron2dTest(unittest.TestCase):
    def test_perceptron_learns_linearly_separable_points(self):
        points = make_training_points()

        history = train_perceptron(points, epochs=8, learning_rate=0.2)
        final_state = history[-1]
        predictions = [
            predict_label(score_point(final_state.weights, final_state.bias, point.features))
            for point in points
        ]

        self.assertEqual(predictions, [point.label for point in points])
        self.assertEqual(final_state.mistakes, 0)

    def test_epoch_state_separates_training_updates_from_current_wrong_predictions(self):
        points = make_training_points()

        history = train_perceptron(points, epochs=1, learning_rate=0.2)
        first_epoch = history[1]

        self.assertGreater(first_epoch.updates, 0)
        self.assertGreater(first_epoch.wrong_predictions, 0)

    def test_training_can_export_visual_steps(self):
        points = make_training_points()

        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "perceptron_steps"

            train_perceptron(points, epochs=2, learning_rate=0.2, output_dir=output_dir)

            expected_files = [
                output_dir / "epoch_00_init.png",
                output_dir / "epoch_01.png",
                output_dir / "epoch_02.png",
                output_dir / "update_steps" / "step_00_init.png",
                output_dir / "update_steps" / "step_01_epoch_01.png",
            ]
            for image_path in expected_files:
                with Image.open(image_path) as image:
                    self.assertEqual(image.format, "PNG")


if __name__ == "__main__":
    unittest.main()
