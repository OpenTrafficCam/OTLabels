from typing import Tuple
from yolov5 import train

import config as CONFIG
from config import load_config
from pathlib import Path

from OTLabels.preprocessing import get_coco_data
from OTLabels.preprocessing import filter_labels
from OTLabels.preprocessing import cvat_to_yolo


def main(config_path):
    config = load_config(config_path)

    # Get COCO dataset
    get_coco_data.main(
        image_urls=config["coco_img_urls"],
        ann_url=config["coco_anns_urls"],
        save_path=config["data_dir"],
        force_download=config["force_download_coco"],
    )

    # Filter COCO dataset
    if config["filter_classes"]:
        filter_labels.main(
            path=config["coco_dir"],
            labels_filter=config["labels_cvat"],
            force_filtering=config["force_filtering_labels"],
        )

    if config["use_custom_datasets"]:
        cvat_to_yolo.main(
            dest_path=config["custom_dir"],
            cvat_path=config["custom_datasets"],
            labels_cvat_path=config["labels_cvat"],
            coco_ann_file_path=config["coco_json_file"],
            new_dataset_name=config["custom_dataset_name"],
        )

    try:
        # Use last.pt as model weight and create another training session
        last_pt, next_model_name = _get_last_pt_path_and_next_model_name(config)
        train.run(
            weights=last_pt,
            cfg=config["model_cfg"],
            data=config["data_config"],
            hyp=config["model_hyp"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            project=config["project_name"],
            name=next_model_name,
        )
    except FileNotFoundError:
        # First run with settings specified in config
        train.run(
            weights=config["model_weights"],
            cfg=config["model_cfg"],
            data=config["data_config"],
            hyp=config["model_hyp"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            project=config["project_name"],
            name=config["model_name"],
        )


def _get_last_pt_path_and_next_model_name(config: dict) -> Tuple[Path, str]:
    wandb_project_dir = Path(config["project_name"])
    model_name = config["model_name"]
    if not wandb_project_dir.exists():
        raise FileNotFoundError(f"Directory [{wandb_project_dir}]doesn't exist!")

    current_idx = len([_dir for _dir in wandb_project_dir.iterdir() if _dir.is_dir()])
    assert current_idx != 0, f"Empty directory! [{wandb_project_dir}]"

    if current_idx == 1:
        last_pt_path = wandb_project_dir / f"{model_name}/weights/last.pt"
    else:
        last_pt_path = wandb_project_dir / f"{model_name}_{current_idx}/weights/last.pt"

    next_model_name = f"{model_name}_{current_idx + 1}"
    assert last_pt_path.is_file(), f"File '{last_pt_path}' doesn't exist!"

    return last_pt_path, next_model_name


if __name__ == "__main__":
    config_path = CONFIG.MODEL_TRAINING_CONFIG_PATH
    main(config_path)
