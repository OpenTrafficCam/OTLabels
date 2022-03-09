from typing import Tuple
from typing import Union
from yolov5 import train

import config as CONFIG
from config import load_config
from pathlib import Path

from OTLabels.preprocessing import get_coco_data
from OTLabels.preprocessing import filter_labels
from OTLabels.preprocessing import cvat_to_yolo


class LastPtNotFoundError(Exception):
    def __init__(
        self, msg: str, project_path: Union[str, Path], model_name: str, index: int
    ) -> None:
        assert index > 0
        self.msg = msg
        self.last_pt_path = project_path
        self.model_name = model_name
        self.index = index


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
            normalized=config["coco_normalized"],
            lower_thresh=config["lower_thresh"],
            upper_thresh=config["upper_thresh"],
            apply_thresh_filter=config["apply_thresh_filter"],
            discard_img_above_thresh=config["discard_img_above_thresh"],
            discard_img_below_thresh=config["discard_img_below_thresh"],
            keep_discarded_imgs=config["keep_discarded_imgs"],
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
    return
    try:
        # Use last.pt as model weight and create another training session
        last_pt, next_model_name = _get_last_pt_and_next_model_name(config)
        train.run(
            weights=last_pt,
            cfg=config["model_cfg"],
            data=config["data_config"],
            hyp=config["model_hyp"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            project=config["project_name"],
            name=next_model_name,
            exist_ok=True,
        )
    except LastPtNotFoundError as lpfnfe:
        try:
            _last_pt = _search_last_pt_recursively(
                lpfnfe.last_pt_path, lpfnfe.model_name, lpfnfe.index
            )

            train.run(
                weights=_last_pt,
                cfg=config["model_cfg"],
                data=config["data_config"],
                hyp=config["model_hyp"],
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                project=config["project_name"],
                name=f"{lpfnfe.model_name}_{lpfnfe.index}",
                exist_ok=True,
            )
        except FileNotFoundError:
            # No last.pt file could be found recursively
            # => start training from scratch
            train.run(
                weights=config["model_weights"],
                cfg=config["model_cfg"],
                data=config["data_config"],
                hyp=config["model_hyp"],
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                project=config["project_name"],
                name=config["model_name"],
                exist_ok=True,
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
            exist_ok=True,
        )


def _search_last_pt_recursively(
    project_path: Union[str, Path], model_name: str, index: int
) -> Path:
    assert index > 0, "index=0 implies empty directory! Violates the pre-condition!"
    last_pt_rel_path = "weights/last.pt"

    if Path(project_path, f"{model_name}_{index}", last_pt_rel_path).exists():
        return Path(project_path, f"{model_name}_{index}", last_pt_rel_path)

    project_dir = Path(project_path)
    if index == 1:
        current_last_pt = project_dir / model_name / last_pt_rel_path
        if not current_last_pt.exists():
            raise FileNotFoundError("No last.pt could be found recursively.")

        return current_last_pt

    if index == 2:
        prev_last_pt = project_dir / model_name / last_pt_rel_path
        if prev_last_pt.exists():
            return prev_last_pt

    return _search_last_pt_recursively(project_dir, model_name, index - 1)


def _get_last_pt_and_next_model_name(config: dict) -> Tuple[Path, str]:
    wandb_project_dir = Path(config["project_name"])
    model_name = config["model_name"]
    if not wandb_project_dir.exists():
        raise FileNotFoundError(f"Directory [{wandb_project_dir}]doesn't exist!")

    current_idx = len([_dir for _dir in wandb_project_dir.iterdir() if _dir.is_dir()])

    if current_idx == 0:
        raise FileNotFoundError(
            f"Directory: '{wandb_project_dir}' is empty."
            + "Start training from scratch!"
        )

    if current_idx == 1:
        last_pt_path = wandb_project_dir / f"{model_name}/weights/last.pt"
    else:
        last_pt_path = wandb_project_dir / f"{model_name}_{current_idx}/weights/last.pt"

    next_model_name = f"{model_name}_{current_idx + 1}"
    if not last_pt_path.exists():
        raise LastPtNotFoundError(
            f"last.pt file not found at: '{last_pt_path}'",
            wandb_project_dir,
            model_name,
            current_idx,
        )

    return last_pt_path, next_model_name


if __name__ == "__main__":
    config_path = CONFIG.MODEL_TRAINING_CONFIG_PATH
    main(config_path)
