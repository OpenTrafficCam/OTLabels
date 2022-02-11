from yolov5 import train

import config as CONFIG
from config import load_config


def main(config_path):
    config = load_config(config_path)
    train.run(
        weights=config["model_weights"],
        cfg=config["model_cfg"],
        data=config["data_config"],
        hyp=config["model_hyp"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        project=config["project_name"],
        name=config["model_name"],
        resume=True,
    )


if __name__ == "__main__":
    config_path = CONFIG.MODEL_TRAINING_CONFIG_PATH
    main(config_path)
