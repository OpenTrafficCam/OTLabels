from ultralytics import YOLO


def train_basic_model():
    model = YOLO("yolov8n.pt")  # pass any model type
    model.train(data="train-config.yaml", epochs=5)


if __name__ == "__main__":
    train_basic_model()
