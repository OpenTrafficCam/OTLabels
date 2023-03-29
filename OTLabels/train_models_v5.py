import torch
from ultralytics import checks

from OTLabels.helpers import machine


def train_basic_model():
    checks()
    model = torch.hub.load(
        "ultralytics/yolov5", "yolov5s", autoshape=False
    )  # pass any model type
    device = get_device()
    model.train(data="config/train-config-prototype.yaml", epochs=1, device=device)


def get_device():
    if machine._has_cuda():
        return "cuda:0"
    return "cpu"


if __name__ == "__main__":
    train_basic_model()
