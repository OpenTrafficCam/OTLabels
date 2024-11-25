import json
from pathlib import Path

import fire

DEFAULT_PATH = (
    Path("/Volumes/platomo data/Produkte/OpenTrafficCam")
    / "OTLabels"
    / "Training"
    / "Daten"
    / "data_mio_svz"
)


def generate(
    output_file: Path,
    project_type: str,
    class_file: Path = Path("OTLabels/config/classes_OTC.json"),
) -> None:
    classifications = load_classifications(class_file)
    all_datasets = generate_dataset_config(classifications, project_type=project_type)

    with open(output_file, mode="w") as output:
        json.dump(all_datasets, output, indent=2)


def generate_dataset_config(
    classifications: list[str],
    project_type: str,
    base_path: Path = DEFAULT_PATH,
) -> dict:
    resolutions = ["720x480", "960x540", "1280x720"]
    all_datasets = {}
    for resolution in resolutions:
        for classification in classifications:
            key, value = create_data(
                classification=classification,
                resolution=resolution,
                project_type=project_type,
                base_path=base_path,
            )
            all_datasets[key] = value
    return all_datasets


def load_classifications(class_file: Path) -> list[str]:
    with open(class_file) as json_file:
        classes = json.load(json_file)
        return [label for label in classes.keys()]


def create_data(
    classification: str,
    resolution: str,
    project_type: str,
    base_path: Path,
) -> tuple[str, dict]:
    name = f"{classification}_{resolution}"
    cam_type = f"mioVision_{resolution}"
    comments = f"SVZ-2024-{classification}"
    image_path = base_path / project_type / resolution / classification
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


if __name__ == "__main__":
    fire.Fire(generate)
