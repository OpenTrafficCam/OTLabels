"""Manage image data using fiftyone"""

import json
from pathlib import Path

import fiftyone
import pandas


class ImportImages:
    def __init__(
        self,
        config_file: str,
        class_file: str = "",
        filter_sites: list = [],
        set_tags: bool = True,
    ) -> None:
        self.set_tags = set_tags

        with open(config_file) as json_file:
            self.config = json.load(json_file)

            if len(filter_sites) > 0:
                self.config = self.config[filter_sites]

        if class_file != "":
            with open(class_file) as json_file:
                self.classes = json.load(json_file)

    def initial_import(
        self,
        import_labels: bool = False,
        launch_app: bool = False,
        name: str = "OTLabels",
        overwrite: bool = False,
    ) -> None:
        if name in fiftyone.list_datasets():
            dataset = fiftyone.load_dataset(name)
            if overwrite:
                dataset.delete()
                dataset = fiftyone.Dataset(name=name, persistent=True)
                print(f"Overwriting Dataset {name}.")
            else:
                print(f"Dataset {name} already exists, loading it from database.")
        else:
            dataset = fiftyone.Dataset(name=name, persistent=True)

        samples = []
        class_dict = {id: label for label, id in self.classes.items()}

        for site in self.config:
            img_dir = Path(self.config[site]["image_path"])
            image_pattern = img_dir.glob("*")

            for img in image_pattern:
                sample = fiftyone.Sample(filepath=img)

                if import_labels:
                    detections = []
                    file_type = Path(img).suffix
                    label_path = (
                        str(img)
                        .replace("images", "labels")
                        .replace(str(file_type), ".txt")
                    )

                    if (
                        Path(label_path).exists()
                        and Path(label_path).stat().st_size > 0
                    ):
                        labels = pandas.read_csv(label_path, sep=" ", header=None)

                        for i in labels.index:
                            label = labels.loc[[i]]
                            label_class = class_dict[label[0].values[0]]

                            bounding_box = [
                                label[1].values[0] - (label[3].values[0] / 2),
                                label[2].values[0] - (label[4].values[0] / 2),
                                label[3].values[0],
                                label[4].values[0],
                            ]

                            detections.append(
                                fiftyone.Detection(
                                    label=label_class, bounding_box=bounding_box
                                )
                            )

                        sample["pre_annotation"] = fiftyone.Detections(
                            detections=detections
                        )

                    else:
                        """with open(label_path, "w") as file:
                        pass"""
                        sample["pre_annotation"] = fiftyone.Detections()

                    tags = self.config[site]["tags"]

                    for tag in tags.keys():
                        sample[tag] = tags[tag]

                    sample["status"] = "pre-annotated"

                else:
                    sample["status"] = "imported"

                samples.append(sample)

        # Create dataset
        dataset.add_samples(samples)

        if launch_app:
            session = fiftyone.launch_app(dataset)
            session.wait()

    def delete_dataset(self, name):
        dataset = fiftyone.load_dataset(name)
        dataset.delete()
