"""用 PyTorch 训练一个最小 LeNet 风格 MNIST 分类器。"""

import argparse
import gzip
import struct
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets


class LeNet(nn.Module):
    """LeNet 风格网络：卷积提取笔画特征，全连接输出 10 个数字分数。"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


def get_device():
    """优先使用 Apple Silicon 的 MPS，其次使用 CUDA，最后回退 CPU。"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def read_mnist_images(path):
    """直接读取 MNIST 原始图片文件，避免当前环境里 torchvision 转 PIL 时依赖 NumPy。"""
    with gzip.open(path, "rb") as file:
        magic, image_count, rows, columns = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"不是 MNIST 图片文件: {path}")

        data = bytearray(file.read())

    images = torch.frombuffer(data, dtype=torch.uint8)
    images = images.reshape(image_count, 1, rows, columns)
    return images.float() / 255.0


def read_mnist_labels(path):
    """直接读取 MNIST 原始标签文件。"""
    with gzip.open(path, "rb") as file:
        magic, label_count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"不是 MNIST 标签文件: {path}")

        data = bytearray(file.read())

    return torch.frombuffer(data, dtype=torch.uint8).long().reshape(label_count)


def make_mnist_dataset(raw_dir, train=True):
    """从 torchvision 下载好的 raw 目录组装 TensorDataset。"""
    if train:
        images_path = raw_dir / "train-images-idx3-ubyte.gz"
        labels_path = raw_dir / "train-labels-idx1-ubyte.gz"
    else:
        images_path = raw_dir / "t10k-images-idx3-ubyte.gz"
        labels_path = raw_dir / "t10k-labels-idx1-ubyte.gz"

    return TensorDataset(read_mnist_images(images_path), read_mnist_labels(labels_path))


def make_dataloaders(data_dir="data", batch_size=64):
    """下载并加载 MNIST。"""
    # 只借用 torchvision 的下载能力，实际读取用上面的 PyTorch 张量版本。
    datasets.MNIST(data_dir, train=True, download=True)
    datasets.MNIST(data_dir, train=False, download=True)

    raw_dir = Path(data_dir) / "MNIST" / "raw"
    train_dataset = make_mnist_dataset(raw_dir, train=True)
    test_dataset = make_mnist_dataset(raw_dir, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader


def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        scores = model(images)
        loss = loss_fn(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(train_loader.dataset)


def evaluate(model, test_loader, device):
    model.eval()
    correct_count = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            scores = model(images)
            predictions = scores.argmax(dim=1)
            correct_count += (predictions == labels).sum().item()

    return correct_count / len(test_loader.dataset)


def train(epochs=5, batch_size=64, learning_rate=0.001, data_dir="data"):
    device = get_device()
    train_loader, test_loader = make_dataloaders(data_dir=data_dir, batch_size=batch_size)
    model = LeNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        accuracy = evaluate(model, test_loader, device)
        print(f"epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.2%}")

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple LeNet-style MNIST model.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--data-dir", default="data")
    return parser.parse_args()


def main():
    args = parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
