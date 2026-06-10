from pathlib import Path
import argparse

import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50


def parse_args():
    parser = argparse.ArgumentParser(description="Run ResNet-50 image classification with PyTorch.")
    parser.add_argument("image", type=Path, help="Path to the image file.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions to print.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    # 加载 ImageNet 预训练权重，并使用与权重匹配的图像预处理流程。
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()

    image = Image.open(args.image).convert("RGB")
    batch = preprocess(image).unsqueeze(0)

    model = resnet50(weights=weights)
    model.eval()

    # 推理阶段不需要梯度，关闭梯度计算可以减少内存占用。
    with torch.no_grad():
        prediction = model(batch).softmax(1)[0]

    scores, class_ids = prediction.topk(args.top_k)
    for rank, (score, class_id) in enumerate(zip(scores, class_ids), start=1):
        category = weights.meta["categories"][class_id.item()]
        print(f"{rank}. {category}: {score.item():.4f}")


if __name__ == "__main__":
    main()
