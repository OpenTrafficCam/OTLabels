"""
Script for retraining a pretrained YOLOv5m model on a custom dataset.

Ideas for model naming convention:
pretrained: Indicates using the pretrained YOLO model for training.
untrained: Indicates a YOLOv5 model that has not been trained.
trained: Indicates on which datasets the model has already been trained on by adding the dataset names after the keyword `trained`.
retrain: Indicates on which datasets the model is going to be trained on.

COCO6cls: Indicates the COCO dataset that has been filtered with a custom set of 6 classes.
COCO: The whole COCO dataset.
custom: Custom dataset.

So an example of a model name could be as follows:

yolov5-s_COCO6cls
yolov5-s_pretrainedCOCO_COCO6cls

yolov5-m_COCO6cls
yolov5-m_pretrainedCOCO_COCO6cls

"""

import wandb
from yolov5 import train 

wandb.login()

# Execute train command