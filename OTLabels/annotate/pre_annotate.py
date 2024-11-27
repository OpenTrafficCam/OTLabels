"""Pre-annotate image data"""

# import modules
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from re import Pattern
from typing import Optional

import pandas
from pandas import DataFrame
from tqdm import tqdm
from ultralytics import YOLO

from OTLabels.annotate.otc_classes import OtcClass
from OTLabels.dataset.generator import ImageDirectory

LABEL_GLOB: str = "*.txt"
IMAGE_GLOB: str = "*.png"
SVZ_IMAGE_PATTERN = re.compile(
    r"(?P<site>[A-Za-z0-9]+)_"
    r"(?P<prefix>[A-Za-z0-9]+)_"
    r"(?P<cameraname>[A-Za-z0-9]+)_"
    r"(?P<year>\d{4})-"
    r"(?P<month>\d{2})-"
    r"(?P<day>\d{2})_"
    r".*"
    r"\.mp4_"
    r"(?P<frame>\d+)"
    r"\.png"
)


@dataclass(frozen=True)
class Image:
    image_directory: ImageDirectory
    image_path: Path
    site: str
    cam_type: str
    frame: int

    def move(self, to: Path) -> None:
        target = self.image_directory.relative_to(to) / self.image_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.image_path), str(target))

    def __str__(self) -> str:
        return str(self.image_path)


def collect_images_in(
    directories: dict[OtcClass, list[ImageDirectory]]
) -> dict[OtcClass, list[Image]]:
    """
    Collects images from specified directories and categorizes them based on the given
    classification.

    Args:
        directories (dict[str, list[Path]]): A dictionary where each key is a
            classification and the value is a list of Path objects representing
            directories to search for images.

    Returns:
        dict[str, list[Image]]: A dictionary where each key is a classification and the
        value is a list of Image objects representing the images found.
    """
    images = defaultdict(list)
    for classification, dirs in directories.items():
        for directory in dirs:
            for image_path in directory.path.glob(IMAGE_GLOB):
                image = _create_from(directory, image_path)
                if image:
                    images[classification].append(image)
    return images


def _create_from(
    directory: ImageDirectory, image_path: Path, pattern: Pattern = SVZ_IMAGE_PATTERN
) -> Optional[Image]:
    match = pattern.match(image_path.name)
    if match:
        site = match.group("site")
        cam_type = match.group("prefix")
        frame = int(match.group("frame"))
        return Image(directory, image_path, site, cam_type, frame)
    return None


def select_images(
    images: dict[OtcClass, list[Image]],
    samples_per_class: dict[OtcClass, int],
    frame_gap=10,
) -> dict[OtcClass, list[Image]]:
    """
    Selects a specified number of images per classification from the provided image
    dictionary. The images will be selected from all sites.

    If the number of samples is less than the number of images, not all sites are
    considered. If the number of samples is greater than the number of images,
    all images will be returned.

    Args:
        images (dict[str, list[Image]]): Dictionary mapping classification labels to
            lists of images.
        samples_per_class (dict[str, int]): Dictionary mapping classification labels to
            the number of images to select per label.
        frame_gap (int): The number of images to skip before selecting an image of the
            same site.

    Returns:
        dict[str, list[Image]]: Dictionary with the same classification labels as keys
            and lists of selected images as values.
    """
    selected_images: dict[OtcClass, list[Image]] = defaultdict(list)
    for classification, images_list in images.items():
        filtered_images = drop_images_too_close(images_list, frame_gap)
        reordered_images = reorder_samples(filtered_images)
        for image in reordered_images:
            if len(selected_images[classification]) < samples_per_class[classification]:
                selected_images[classification].append(image)
            else:
                break
    return selected_images


def drop_images_too_close(images: list[Image], frame_gap: int) -> list[Image]:
    sorted_images = sort_images(images)
    last_image = sorted_images[0]
    filtered_images = [last_image]
    for image in sorted_images[1:]:
        site_changed = image.site != last_image.site
        frmae_difference_large_enough = image.frame - last_image.frame > frame_gap
        if site_changed or frmae_difference_large_enough:
            filtered_images.append(image)
            last_image = image
    return filtered_images


def sort_images(images: list[Image]) -> list[Image]:
    return sorted(images, key=lambda x: (x.site, x.frame))


def reorder_samples(images: list[Image]) -> list[Image]:
    """
    This method reorders the given samples by taking one frame per site before adding
    the next frame of the site. The site a frame was taken changes in the resulting list
    between two consecutive frames, if there are frames available of at least two site.

    Args:
        images (list[Image]): Images to reorder

    Returns: list of images

    """
    dataset_ordered = []

    samples = defaultdict(list)
    sorted_images = sort_images(images)
    sorted_images.reverse()
    for image in sorted_images:
        samples[image.site].append(image)

    sites = sorted(samples.keys())
    finished_sites: set = set()
    while sites != sorted(finished_sites):
        for site in sites:
            try:
                element = samples[site].pop()
                dataset_ordered.append(element)
            except IndexError:
                finished_sites.add(site)
    return dataset_ordered


def move_images(images: dict[OtcClass, list[Image]], output_path: Path) -> None:
    for image_list in images.values():
        for image in image_list:
            image.move(to=output_path)


class PreAnnotateImages:
    def __init__(
        self,
        config_file: str | None = None,
        config: dict | None = None,
        class_file: str = "",
        filter_sites: list = [],
        model_file: str = "yolov8n.pt",
    ) -> None:
        self.model = YOLO(model_file, task="detect")

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

    def _filter_classes(self, site, label_dir) -> None:
        classes = self.classes
        annotation_files = [str(file) for file in label_dir.glob(LABEL_GLOB)]

        print(
            f"Filter labels in {label_dir} by class "
            + ", ".join(str(e) for e in list(classes.keys()))
            + "..."
        )

        for annotation_file in tqdm(annotation_files):
            if Path(annotation_file).stat().st_size <= 0:
                continue
            file_labels: DataFrame = pandas.read_csv(
                annotation_file,
                header=None,
                sep=" ",
            )
            file_labels = file_labels[file_labels[0].isin(classes.values())]
            file_labels.to_csv(
                path_or_buf=Path(annotation_file),
                header=False,
                sep=" ",
                index=False,
                lineterminator="\n",
            )

    def pre_annotate(self) -> None:
        for site in self.config:
            label_dir = Path(self.config[site]["label_path"] + "/pre_annotation_labels")
            labels = label_dir.glob(LABEL_GLOB)
            n = 0
            for label in labels:
                Path.unlink(label)
                n = +1

            if n > 0:
                print("{} label(s) deleted!".format(n))

            predictor = self.model
            """
            This is a temporal fix to initialize the predictor,
            so the output directory for the labels can be edited.
            """
            # TODO: edit output directory in a smarter way
            img_dir = Path(self.config[site]["image_path"])
            image_pattern = img_dir.glob(IMAGE_GLOB)
            try:
                first_image = next(image_pattern)
                predictor(source=first_image)
                Path.unlink(Path(first_image))
                self.model.predictor.save_dir = Path(self.config[site]["image_path"])
                predictor(
                    source=self.config[site]["image_path"],
                    save_txt=True,
                    agnostic_nms=True,
                )

                if self.classes:
                    self._filter_classes(site, label_dir)
            except StopIteration:
                continue


@dataclass(frozen=True)
class User:
    open_project: str
    cvat: str
