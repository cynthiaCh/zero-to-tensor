from pathlib import Path
import argparse

import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50


def parse_args():
    parser = argparse.ArgumentParser(description="Explain ResNet-50 layers with PyTorch.")
    parser.add_argument("image", type=Path, help="Path to the image file.")
    return parser.parse_args()


def print_stage_guide():
    print("ResNet-50 high-level stages:")
    print("conv1    : 第一层大卷积，先提取低级边缘/颜色/纹理特征")
    print("bn1/relu : 批归一化和非线性激活，让训练与表达更稳定")
    print("maxpool  : 下采样，降低特征图尺寸")
    print("layer1   : 残差块组 1，输出 256 通道，保留较细空间信息")
    print("layer2   : 残差块组 2，输出 512 通道，继续下采样")
    print("layer3   : 残差块组 3，输出 1024 通道，提取更抽象语义")
    print("layer4   : 残差块组 4，输出 2048 通道，形成高级语义特征")
    print("avgpool  : 全局平均池化，把空间特征压成向量")
    print("fc       : 全连接分类层，输出 ImageNet 1000 类分数")
    print()


def main():
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    model = resnet50(weights=weights)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    batch = preprocess(image).unsqueeze(0)

    stage_names = [
        "conv1",
        "bn1",
        "relu",
        "maxpool",
        "layer1",
        "layer2",
        "layer3",
        "layer4",
        "avgpool",
        "fc",
    ]

    hooks = []

    def make_hook(name):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                print(f"{name:<8} output shape: {tuple(output.shape)}")

        return hook

    for name in stage_names:
        hooks.append(getattr(model, name).register_forward_hook(make_hook(name)))

    print_stage_guide()
    print(f"input    output shape: {tuple(batch.shape)}")

    # 推理时只看前向传播，不需要计算梯度。
    with torch.no_grad():
        logits = model(batch)
        probabilities = logits.softmax(1)[0]

    for hook in hooks:
        hook.remove()

    print()
    print("Top-5 predictions:")
    scores, class_ids = probabilities.topk(5)
    for rank, (score, class_id) in enumerate(zip(scores, class_ids), start=1):
        category = weights.meta["categories"][class_id.item()]
        print(f"{rank}. {category}: {score.item():.4f}")


if __name__ == "__main__":
    main()
