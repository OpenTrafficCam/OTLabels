from pathlib import Path

WORKING_DIR = Path(__file__).parent

# Training Parameters
BATCH_SIZE = 32
EPOCHS = 1

# Model Configuration
MODEL_WEIGHTS = "yolov5m.pt"
MODEL_NAME = "yolov5-m_COCO6cls"
MODEL_CFG = Path(WORKING_DIR, "OTLabels/models/yolov5m_6cl.yaml")
MODEL_HYP = Path(WORKING_DIR, "OTLabels/data/hyp.finetune.yaml")
PROJECT_NAME = "OTLabels"


# Data Configuration
DOWNLOAD_COCO = True
USE_COCO = True
FILTER_CLASSES = True
DATA_DIR = Path(WORKING_DIR, "OTLabels/data")
DATA_CONFIG = Path(WORKING_DIR, "data/coco_6cl.yaml")
COCO_IMAGE_URLS = Path(WORKING_DIR, "OTLabels/coco_image_URLs.txt")
COCO_ANNS_URLS = Path(WORKING_DIR, "OTLabels/coco_annotation_URLs.txt")
COCO_DIR = Path(WORKING_DIR, "OTLabels/data/coco")
LABELS_FILTER = Path(WORKING_DIR, "OTLabels/labels_CVAT.txt")
