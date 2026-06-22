"""二维感知机训练演示：从一条会移动的分类线理解权重更新。"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class TrainingPoint:
    """一个二维训练样本：features 是输入，label 是正确类别。"""

    features: tuple[float, float]
    label: int


@dataclass(frozen=True)
class PerceptronState:
    """记录某个 epoch 后的参数，便于观察训练过程。"""

    epoch: int
    weights: tuple[float, float]
    bias: float
    updates: int
    wrong_predictions: int

    @property
    def mistakes(self):
        """兼容旧测试：mistakes 表示当前还分错的点数。"""
        return self.wrong_predictions


def make_training_points():
    """构造一组能被直线分开、但需要多轮训练的二维点。"""
    return [
        TrainingPoint((-2.0, -2.0), 0),
        TrainingPoint((-1.0, 2.0), 0),
        TrainingPoint((2.0, -1.0), 0),
        TrainingPoint((-2.0, 1.0), 0),
        TrainingPoint((1.0, 1.0), 1),
        TrainingPoint((2.0, 2.0), 1),
        TrainingPoint((3.0, 0.0), 1),
        TrainingPoint((0.0, 3.0), 1),
    ]


def score_point(weights, bias, features):
    """计算感知机分数：w1*x1 + w2*x2 + b。"""
    x1, x2 = features
    w1, w2 = weights
    return w1 * x1 + w2 * x2 + bias


def predict_label(score):
    """分数大于等于 0 判为 1，否则判为 0。"""
    return 1 if score >= 0 else 0


def update_perceptron(weights, bias, point, learning_rate):
    """预测错时按感知机规则更新权重和偏置。"""
    prediction = predict_label(score_point(weights, bias, point.features))
    error = point.label - prediction
    if error == 0:
        return weights, bias, False

    x1, x2 = point.features
    w1, w2 = weights
    updated_weights = (
        w1 + learning_rate * error * x1,
        w2 + learning_rate * error * x2,
    )
    updated_bias = bias + learning_rate * error
    return updated_weights, updated_bias, True


def count_wrong_predictions(points, weights, bias):
    """统计当前参数下还有多少点会被分错。"""
    wrong_count = 0
    for point in points:
        prediction = predict_label(score_point(weights, bias, point.features))
        if prediction != point.label:
            wrong_count += 1
    return wrong_count


def train_perceptron(points, epochs=8, learning_rate=0.1, output_dir=None):
    """训练二维感知机，并可选导出每个 epoch 后的分类线图片。"""
    weights = (0.0, 0.0)
    bias = 0.0
    step = 0
    history = [
        PerceptronState(
            epoch=0,
            weights=weights,
            bias=bias,
            updates=0,
            wrong_predictions=count_wrong_predictions(points, weights, bias),
        )
    ]

    if output_dir:
        output_dir = Path(output_dir)
        draw_state(points, history[-1], output_dir / "epoch_00_init.png")
        draw_state(points, history[-1], output_dir / "update_steps" / "step_00_init.png")

    for epoch in range(1, epochs + 1):
        updates = 0
        for point in points:
            weights, bias, changed = update_perceptron(weights, bias, point, learning_rate)
            if changed:
                updates += 1
                step += 1
                if output_dir:
                    step_state = PerceptronState(
                        epoch=epoch,
                        weights=weights,
                        bias=bias,
                        updates=updates,
                        wrong_predictions=count_wrong_predictions(points, weights, bias),
                    )
                    draw_state(
                        points,
                        step_state,
                        output_dir / "update_steps" / f"step_{step:02d}_epoch_{epoch:02d}.png",
                    )

        state = PerceptronState(
            epoch=epoch,
            weights=weights,
            bias=bias,
            updates=updates,
            wrong_predictions=count_wrong_predictions(points, weights, bias),
        )
        history.append(state)
        if output_dir:
            draw_state(points, state, output_dir / f"epoch_{epoch:02d}.png")

    return history


def to_canvas(point, width, height, padding, axis_min=-3.0, axis_max=3.0):
    """把数学坐标转换成图片坐标。"""
    x, y = point
    scale_x = (width - padding * 2) / (axis_max - axis_min)
    scale_y = (height - padding * 2) / (axis_max - axis_min)
    canvas_x = padding + (x - axis_min) * scale_x
    canvas_y = height - padding - (y - axis_min) * scale_y
    return canvas_x, canvas_y


def decision_boundary_points(weights, bias, axis_min=-3.0, axis_max=3.0):
    """根据 w1*x + w2*y + b = 0 计算分类线的两个端点。"""
    w1, w2 = weights
    if abs(w2) > 1e-9:
        x_left = axis_min
        y_left = -(w1 * x_left + bias) / w2
        x_right = axis_max
        y_right = -(w1 * x_right + bias) / w2
        return (x_left, y_left), (x_right, y_right)

    if abs(w1) > 1e-9:
        x = -bias / w1
        return (x, axis_min), (x, axis_max)

    return None


def draw_state(points, state, output_path):
    """画出二维点、当前分类线和参数值。"""
    width = 640
    height = 520
    padding = 70
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    axis_color = (180, 180, 180)
    x_axis_start = to_canvas((-3, 0), width, height, padding)
    x_axis_end = to_canvas((3, 0), width, height, padding)
    y_axis_start = to_canvas((0, -3), width, height, padding)
    y_axis_end = to_canvas((0, 3), width, height, padding)
    draw.line([x_axis_start, x_axis_end], fill=axis_color, width=1)
    draw.line([y_axis_start, y_axis_end], fill=axis_color, width=1)

    for point in points:
        x, y = to_canvas(point.features, width, height, padding)
        color = (220, 60, 60) if point.label == 0 else (50, 120, 220)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline=(30, 30, 30))
        prediction = predict_label(score_point(state.weights, state.bias, point.features))
        if prediction != point.label:
            draw.rectangle((x - 11, y - 11, x + 11, y + 11), outline=(20, 20, 20), width=2)

    boundary = decision_boundary_points(state.weights, state.bias)
    if boundary:
        start = to_canvas(boundary[0], width, height, padding)
        end = to_canvas(boundary[1], width, height, padding)
        draw.line([start, end], fill=(20, 20, 20), width=3)

    w1, w2 = state.weights
    draw.text((20, 18), f"epoch: {state.epoch}", fill=(30, 30, 30))
    draw.text((20, 40), f"w1={w1:.2f}, w2={w2:.2f}, b={state.bias:.2f}", fill=(30, 30, 30))
    draw.text((20, 62), f"updates this epoch: {state.updates}", fill=(30, 30, 30))
    draw.text((20, 84), f"wrong now: {state.wrong_predictions}", fill=(30, 30, 30))
    draw.text((20, height - 36), "red=0, blue=1, box=wrong prediction", fill=(80, 80, 80))
    image.save(output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a tiny 2D perceptron.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--output-dir", default="outputs/perceptron_2d")
    return parser.parse_args()


def main():
    args = parse_args()
    points = make_training_points()
    history = train_perceptron(
        points,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )

    for state in history:
        w1, w2 = state.weights
        print(
            f"epoch {state.epoch}: "
            f"w1={w1:.2f}, w2={w2:.2f}, b={state.bias:.2f}, "
            f"updates={state.updates}, wrong_now={state.wrong_predictions}"
        )
    print(f"visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
