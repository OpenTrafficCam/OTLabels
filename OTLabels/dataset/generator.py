import json
from pathlib import Path

import fire


def generate(
    output_file: Path, class_file: Path = Path("OTLabels/config/classes_OTC.json")
) -> None:
    resolutions = ["720x480", "960x540", "1280x720"]
    classifications = load_classifications(class_file)
    all_datasets = {}
    for resolution in resolutions:
        for classification in classifications:
            key, value = create_data(
                classification=classification,
                resolution=resolution,
            )
            all_datasets[key] = value

    with open(output_file, mode="w") as output:
        json.dump(all_datasets, output, indent=2)


def load_classifications(class_file: Path) -> list[str]:
    with open(class_file) as json_file:
        classes = json.load(json_file)
        return [label for label in classes.keys()]


def create_data(classification: str, resolution: str) -> tuple[str, dict]:
    name = f"{classification}_{resolution}"
    cam_type = f"mioVision_{resolution}"
    comments = f"SVZ-2024-{classification}"
    image_path = (
        Path("/Volumes/platomo data/Produkte/OpenTrafficCam")
        / "OTLabels"
        / "Training"
        / "Daten"
        / "data_mio_svz"
        / resolution
        / classification
    )
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
