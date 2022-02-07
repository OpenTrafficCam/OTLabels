from pathlib import Path

WORKING_DIR = Path(__file__).parent

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
