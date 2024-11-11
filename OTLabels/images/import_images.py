"""Manage image data using fiftyone"""

import json
from collections import defaultdict
from pathlib import Path
from re import compile

import fiftyone
import pandas


def reorder_samples(samples: dict[str, list]) -> list:
    """
    This method reorders the given samples by taking one frame per site before adding
    the next frame of the site. The site a frame was taken changes in the resulting list
    between two consecutive frames, if there are frames available of at least two site.

    Args:
        samples (dict[str, list]): Dictionary of samples per site

    Returns: list of samples

    """
    dataset_ordered = []
    sites = sorted(samples.keys())
    finished_keys: set = set()
    while sites != sorted(finished_keys):
        for site in sites:
            try:
                element = samples[site].pop()
                dataset_ordered.append(element)
            except IndexError:
                finished_keys.add(site)
    return dataset_ordered


class ImportImages:
    def __init__(
        self,
        config_file: str | None = None,
        config: dict | None = None,
        class_file: str = "",
        filter_sites: list = [],
        set_tags: bool = True,
        otc_pipeline_import: bool = True,
    ) -> None:
        self.set_tags = set_tags
        self.otc_pipeline_import = otc_pipeline_import

        if config:
            self.config = config
        else:
            if config_file is None:
                raise FileNotFoundError()
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
        dataset_name: str = "OTLabels",
        overwrite: bool = False,
    ) -> None:
        if dataset_name in fiftyone.list_datasets():
            dataset = fiftyone.load_dataset(dataset_name)
            if overwrite:
                dataset.delete()
                dataset = fiftyone.Dataset(name=dataset_name, persistent=True)
                print(f"Overwriting Dataset {dataset_name}.")
            else:
                print(
                    f"Dataset {dataset_name} already exists, loading it from database."
                )
        else:
            dataset = fiftyone.Dataset(name=dataset_name, persistent=True)

        samples = defaultdict(list)
        class_dict = {id: label for label, id in self.classes.items()}

        for site in self.config:
            img_dir = Path(self.config[site]["image_path"])
            label_dir = Path(self.config[site]["label_path"])
            image_pattern = img_dir.glob("*.png")

            for img in image_pattern:
                sample = fiftyone.Sample(filepath=img)

                if import_labels:
                    detections = []
                    label_path = (label_dir / img.name).with_suffix(".txt")

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
                    tags = self._update_tags(tags, img)
                    for key, value in tags.items():
                        sample[key] = value

                    sample["status"] = "pre-annotated"

                else:
                    sample["status"] = "imported"

                samples[sample["site"]].append(sample)

        dataset_ordered = reorder_samples(samples)
        # Create dataset
        dataset.add_samples(dataset_ordered)

        if launch_app:
            session = fiftyone.launch_app(dataset)
            session.wait()

            # TODO wozu wird hier fiftyone gestartet?

    def _update_tags(self, tags: dict, img: Path) -> dict:
        if self.otc_pipeline_import:
            pattern = compile(
                r"(?P<site>[A-Za-z0-9]+)_"
                r"(?P<prefix>[A-Za-z0-9]+)_"
                r"(?P<cameraname>[A-Za-z0-9]+)_"
                r"(?P<year>\d{4})-"
                r"(?P<month>\d{2})-"
                r"(?P<day>\d{2})_"
                r".*"
            )
            match = pattern.match(img.name)
            if match:
                tags["site"] = match.group("site")
                tags["cam_type"] = match.group("prefix")
        return tags

    def delete_dataset(self, name):
        dataset = fiftyone.load_dataset(name)
        dataset.delete()
