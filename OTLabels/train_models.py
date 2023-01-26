from ultralytics import YOLO, checks

from OTLabels.helpers import machine


def train_basic_model():
    checks()
    model = YOLO("models/yolov8n.pt")  # pass any model type
    device = get_device()
    model.train(data="config/train-config.yaml", epochs=1, device=device)


def get_device():
    if machine._has_cuda():
        return "cuda:0"
    return "cpu"


if __name__ == "__main__":
    train_basic_model()
