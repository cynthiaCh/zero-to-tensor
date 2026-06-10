from pathlib import Path
import unittest

import torch
from torchvision.models import ResNet50_Weights

from experiments import train_resnet_custom as trainer


class TrainResnetCustomTest(unittest.TestCase):
    def test_create_transforms_uses_training_augmentation_and_validation_preprocessing(self):
        train_transform, val_transform = trainer.create_transforms(ResNet50_Weights.DEFAULT)

        train_steps = [type(step).__name__ for step in train_transform.transforms]
        val_steps = [type(step).__name__ for step in val_transform.transforms]

        self.assertIn("RandomResizedCrop", train_steps)
        self.assertIn("RandomHorizontalFlip", train_steps)
        self.assertEqual(val_steps, ["Resize", "CenterCrop", "ToTensor", "Normalize"])

    def test_build_model_replaces_classifier_for_custom_class_count(self):
        model = trainer.build_model(num_classes=3, pretrained=False)

        self.assertEqual(model.fc.out_features, 3)

    def test_resolve_device_prefers_requested_cpu(self):
        device = trainer.resolve_device("cpu")

        self.assertEqual(device, torch.device("cpu"))

    def test_parse_args_accepts_imagefolder_training_contract(self):
        args = trainer.parse_args(
            [
                "--data-dir",
                str(Path("data")),
                "--epochs",
                "2",
                "--batch-size",
                "4",
                "--output",
                "checkpoints/model.pt",
            ]
        )

        self.assertEqual(args.data_dir, Path("data"))
        self.assertEqual(args.epochs, 2)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.output, Path("checkpoints/model.pt"))


if __name__ == "__main__":
    unittest.main()
