from yolov5 import train


def train_basic_model():
    train.run(data="config/train-config-prototype.yaml")


if __name__ == "__main__":
    train_basic_model()
