"""Pre-annotate image data"""

# import modules
import json
from pathlib import Path

from ultralytics import YOLO


class PreAnnotateImages:
    def __init__(
        self,
        config_file: str,
        label_file: str,
        filter_sites: list = [],
        model_file: str = "yolov8n.pt",
    ) -> None:
        self.model = YOLO(model_file)

        with open(config_file) as json_file:
            self.config = json.load(json_file)

            if len(filter_sites) > 0:
                self.config = self.config[filter_sites]

        with open(label_file) as json_file:
            self.labels = json.load(json_file)

    def pre_annotate(self) -> None:
        for site in self.config:
            predictor = self.model
            predictor(source="https://ultralytics.com/images/bus.jpg")
            self.model.predictor.save_dir = Path(self.config[site]["label_path"])
            predictor(source=self.config[site]["image_path"], save_txt=True)


PreAnnotateImages(
    config_file="OTLabels/config/training_data.json",
    label_file="OTLabels/config/labels_COCO.json",
).pre_annotate()
