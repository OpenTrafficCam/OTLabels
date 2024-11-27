import json
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import fire


class SampleType(StrEnum):
    CORRECT_CLASSIFICATION = "correct_classification"
    INCORRECT_CLASSIFICATION = "incorrect_classification"


DEFAULT_PATH = (
    Path("/Volumes/platomo data/Produkte/OpenTrafficCam")
    / "OTLabels"
    / "Training"
    / "Daten"
    / "data_mio_svz"
)


@dataclass(frozen=True)
class ImageDirectory:
    base_path: Path
    sample_type: SampleType
    resolution: str
    classification: str

    @property
    def path(self) -> Path:
        return self.base_path / self.sample_type / self.resolution / self.classification


def generate(
    output_file: Path,
    sample_type: SampleType,
    class_file: Path = Path("OTLabels/config/classes_OTC.json"),
) -> None:
    classifications = load_classifications(class_file)
    all_datasets = generate_dataset_config(classifications, sample_type=sample_type)

    with open(output_file, mode="w") as output:
        json.dump(all_datasets, output, indent=2)


def generate_dataset_config(
    classifications: list[str],
    sample_type: SampleType,
    base_path: Path = DEFAULT_PATH,
) -> dict:
    resolutions = ["720x480", "960x540", "1280x720"]
    all_datasets = {}
    for resolution in resolutions:
        for classification in classifications:
            key, value = create_data(
                classification=classification,
                resolution=resolution,
                sample_type=sample_type,
                base_path=base_path,
            )
            all_datasets[key] = value
    return all_datasets


def generate_image_directories(
    classifications: list[str],
    sample_type: SampleType,
    base_path: Path = DEFAULT_PATH,
) -> dict[str, list[ImageDirectory]]:
    resolutions = ["720x480", "960x540", "1280x720"]
    all_datasets: dict[str, list[ImageDirectory]] = defaultdict(list)
    for resolution in resolutions:
        for classification in classifications:
            image_directory = create_image_path(
                base_path, classification, sample_type, resolution
            )
            all_datasets[classification].append(image_directory)
    return all_datasets


def load_classifications(class_file: Path) -> list[str]:
    with open(class_file) as json_file:
        classes = json.load(json_file)
        return [label for label in classes.keys()]


def create_data(
    classification: str,
    resolution: str,
    sample_type: SampleType,
    base_path: Path,
) -> tuple[str, dict]:
    name = f"{classification}_{resolution}"
    cam_type = f"mioVision_{resolution}"
    comments = f"SVZ-2024-{classification}"
    image_path = create_image_path(
        base_path, classification, sample_type, resolution
    ).path
    label_path = image_path / "labels"
    return name, {
        "tags": {
            "site": "",
            "cam_type": cam_type,
            "recording_start_date": "",
            "recording_end_date": "",
            "lense_type": "normal",
            "geolocation": {"lat": "", "lon": ""},
            "height": "",
            "occasion": "",
            "video_file": "",
            "comments": comments,
        },
        "image_path": str(image_path),
        "label_path": str(label_path),
    }


def create_image_path(
    base_path: Path,
    classification: str,
    sample_type: SampleType,
    resolution: str,
) -> ImageDirectory:
    """
    Create image path for dataset according to the systematically sampled images from
    videos processed for SVZ.

    Args:
        base_path (Path): Base directory to load images from
        classification (str): Classification contained in the image
        sample_type (SampleType): Strategy used to sample images
        resolution (str): Video resolution

    Returns:

    """
    return ImageDirectory(
        base_path=base_path,
        sample_type=sample_type,
        resolution=resolution,
        classification=classification,
    )


if __name__ == "__main__":
    fire.Fire(generate)
