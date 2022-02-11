import yaml
from yaml.loader import SafeLoader
from pathlib import Path

WORKING_DIR = Path(__file__).parent
MODEL_TRAINING_CONFIG_PATH = Path(WORKING_DIR, "configs/yolov5-s_COCO6cls.yaml")
BASE_YOLO_MODELS = ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"]


def load_config(config_yaml_path):
    with open(config_yaml_path) as f:
        config = yaml.load(f, Loader=SafeLoader)

    if config["paths_are_relative"]:
        if not _is_yolo_base_model(config["model_weights"]):
            config["model_weights"] = Path(WORKING_DIR, config["model_weights"])

        config["model_cfg"] = Path(WORKING_DIR, config["model_cfg"])
        config["model_hyp"] = Path(WORKING_DIR, config["model_hyp"])
        config["project_name"] = Path(WORKING_DIR, config["project_name"])

        config["data_dir"] = Path(WORKING_DIR, config["data_dir"])
        config["data_config"] = Path(WORKING_DIR, config["data_config"])
        config["labels_cvat"] = Path(WORKING_DIR, config["labels_cvat"])

        config["coco_img_urls"] = Path(WORKING_DIR, config["coco_img_urls"])
        config["coco_anns_urls"] = Path(WORKING_DIR, config["coco_anns_urls"])
        config["coco_dir"] = Path(WORKING_DIR, config["coco_dir"])
        config["coco_json_file"] = Path(WORKING_DIR, config["coco_json_file"])

        config["custom_dir"] = Path(WORKING_DIR, config["custom_dir"])
        config["custom_datasets"] = Path(WORKING_DIR, config["custom_datasets"])

    return config


def _is_yolo_base_model(model_path: str):
    return model_path.split(".")[0] in BASE_YOLO_MODELS
