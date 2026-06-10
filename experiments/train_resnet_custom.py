from pathlib import Path
import argparse
import copy
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet50_Weights, resnet50


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train ResNet-50 on a custom ImageFolder dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Dataset root. Expected subdirectories: train/<class> and val/<class>.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Training device. auto prefers cuda, then mps, then cpu.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/resnet50_custom.pt"),
        help="Path used to save the best checkpoint.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from random initialization instead of ImageNet weights.",
    )
    return parser.parse_args(argv)


def resolve_device(requested_device):
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "cuda":
        return torch.device("cuda")
    if requested_device == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_transforms(weights):
    normalize = transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std,
    )

    # 训练集加入随机裁剪和翻转，验证集保持确定性预处理，便于对比指标。
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, val_transform


def create_dataloaders(data_dir, batch_size, num_workers, weights):
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    train_transform, val_transform = create_transforms(weights)
    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=train_transform),
        "val": datasets.ImageFolder(val_dir, transform=val_transform),
    }
    dataloaders = {
        "train": DataLoader(
            image_datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            image_datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
    return dataloaders, image_datasets


def build_model(num_classes, pretrained=True):
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    # 自定义数据集类别数通常不同于 ImageNet，需要替换最后的全连接分类头。
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    dataset_size = len(dataloader.dataset)
    return running_loss / dataset_size, running_corrects / dataset_size


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

    dataset_size = len(dataloader.dataset)
    return running_loss / dataset_size, running_corrects / dataset_size


def save_checkpoint(output_path, model, class_names, best_acc, args):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "best_acc": best_acc,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
        },
        output_path,
    )


def train_model(model, dataloaders, criterion, optimizer, device, epochs):
    best_model_state = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, dataloaders["val"], criterion, device)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"time={elapsed:.1f}s"
        )

    model.load_state_dict(best_model_state)
    return model, best_acc


def main():
    args = parse_args()
    device = resolve_device(args.device)
    pretrained = not args.no_pretrained
    weights = ResNet50_Weights.DEFAULT

    dataloaders, image_datasets = create_dataloaders(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        weights,
    )
    class_names = image_datasets["train"].classes
    model = build_model(num_classes=len(class_names), pretrained=pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Classes: {class_names}")
    print(f"Device: {device}")
    model, best_acc = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        device,
        args.epochs,
    )
    save_checkpoint(args.output, model, class_names, best_acc, args)
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Saved checkpoint to: {args.output}")


if __name__ == "__main__":
    main()
