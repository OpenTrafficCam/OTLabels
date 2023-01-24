"""Pre-annotate image data"""

# import modules
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


class PreAnnotateImages:
    def __init__(
        self,
        config_file: str,
        class_file: str = "",
        filter_sites: list = [],
        model_file: str = "yolov8n.pt",
    ) -> None:
        self.model = YOLO(model_file)

        with open(config_file) as json_file:
            self.config = json.load(json_file)

            if len(filter_sites) > 0:
                self.config = self.config[filter_sites]

        if class_file != "":
            with open(class_file) as json_file:
                self.classes = json.load(json_file)

    def _filter_classes(self, site, label_dir) -> None:

        classes = self.classes
        ann_files = [str(file) for file in label_dir.glob("*.txt")]

        print(
            f"Filter labels in {label_dir} by class "
            + ", ".join(str(e) for e in list(classes.keys()))
            + "..."
        )

        for ann_file in tqdm(ann_files):

            if Path(ann_file).stat().st_size > 0:
                file_labels = pd.read_csv(ann_file, header=None, sep=" ")
            else:
                continue

            file_labels = file_labels[file_labels[0].isin(classes.values())]

            file_labels.to_csv(
                ann_file,
                header=False,
                sep=" ",
                index=False,
                line_terminator="\n",
            )

    def pre_annotate(self) -> None:
        for site in self.config:
            label_dir = Path(self.config[site]["label_path"] + "/labels")
            labels = label_dir.glob("*.txt")
            n = 0
            for label in labels:
                Path.unlink(label)
                n = +1

            if n > 0:
                print("{} label(s) deleted!".format(n))

            predictor = self.model
            """
            This is a temporal fix to inizialize the predictor,
            so the output directory for the labels can be edited.
            """
            # TODO: edit output directory in a smarter way
            predictor(source="https://ultralytics.com/images/bus.jpg")
            Path.unlink(Path("./bus.jpg"))
            self.model.predictor.save_dir = Path(self.config[site]["label_path"])
            predictor(source=self.config[site]["image_path"], save_txt=True)

            if self.classes:
                self._filter_classes(site, label_dir)
