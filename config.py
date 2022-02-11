import yaml
from yaml.loader import SafeLoader
from pathlib import Path

WORKING_DIR = Path(__file__).parent
MODEL_TRAINING_CONFIG_PATH = Path(WORKING_DIR, "configs/yolov5-s_COCO.yaml")
BASE_YOLO_MODELS = ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"]

<<<<<<< HEAD
# Training Parameters
BATCH_SIZE = -1  # -1 for auto-batch
EPOCHS = 300
RESUME_TRAINING = True
"""Path to model weights(model.pt) or `True` to fetch latest run."""

# Model Configuration
MODEL_WEIGHTS = ""
"""
Set to model name to use pretrained weights. To train a model from scratch set
`MODEL_WEIGHTS = ""` and `MODEL_CFG = "path/to/model_cfg.yaml"`.
"""
MODEL_NAME = "yolov5-s_COCO"
MODEL_CFG = Path(WORKING_DIR, "yolov5/models/yolov5s.yaml")
MODEL_HYP = Path(WORKING_DIR, "OTLabels/data/hyp.finetune.yaml")
PROJECT_NAME = Path(WORKING_DIR, "OTLabels/data/runs")

# Data Configuration

# FLAGS
FORCE_DOWNLOAD_COCO = False
FORCE_FILTERING_LABELS = False
USE_COCO = True
USE_CUSTOM_DATSETS = False
FILTER_CLASSES = False

DATA_DIR = Path(WORKING_DIR, "OTLabels/data")
DATA_CONFIG = Path(WORKING_DIR, "OTLabels/data/coco.yaml")
LABELS_CVAT = Path(WORKING_DIR, "OTLabels/labels_CVAT.txt")

# COCO Paths
COCO_IMAGE_URLS = Path(WORKING_DIR, "OTLabels/coco_image_URLs.txt")
COCO_ANNS_URLS = Path(WORKING_DIR, "OTLabels/coco_annotation_URLs.txt")
COCO_DIR = Path(WORKING_DIR, "OTLabels/data/coco")
COCO_JSON_FILE = Path(
    WORKING_DIR, "OTLabels/data/coco/annotations/instances_val2017.json"
)

# Custom Datasets
CUSTOM_DIR = Path(WORKING_DIR, "OTLabels/data/custom")
CUSTOM_DATASET_NAME = "otc_custom"
CUSTOM_DATASETS = CUSTOM_DIR
=======

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
>>>>>>> rework-otlabels
