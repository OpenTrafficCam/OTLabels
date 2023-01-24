"""Preprocess image data for annotation in CVAT"""

from annotate.pre_annotate import PreAnnotateImages

PreAnnotateImages(
    config_file="OTLabels/config/training_data.json",
    class_file="OTLabels/config/classes_COCO.json",
).pre_annotate()
