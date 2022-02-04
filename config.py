from pathlib import Path

WORKING_DIR = Path(__file__).parent

# Training Parameters
BATCH_SIZE = 32
EPOCHS = 1
RESUME_TRAINING = True

# Model Configuration
MODEL_WEIGHTS = "yolov5m.pt"
MODEL_NAME = "yolov5-m_COCO6cls"
MODEL_CFG = Path(WORKING_DIR, "OTLabels/models/yolov5m_6cl.yaml")
MODEL_HYP = Path(WORKING_DIR, "OTLabels/data/hyp.finetune.yaml")
PROJECT_NAME = "OTLabels"


# Data Configuration

# FLAGS
FORCE_DOWNLOAD_COCO = False
FORCE_FILTERING_LABELS = False
USE_COCO = True
USE_CUSTOM_DATSETS = False
FILTER_CLASSES = True

DATA_DIR = Path(WORKING_DIR, "OTLabels/data")
DATA_CONFIG = Path(WORKING_DIR, "OTLabels/data/coco_6cl.yaml")
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
