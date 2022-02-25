from typing import Tuple

from yolov5 import train
from pathlib import Path

import config as CONFIG
from config import load_config


def main(config_path, project_path=None, model_name=None, last_pt_path=None):
    config = load_config(config_path)
    try:
        # Use last.pt as model weight and create another training session
        if project_path is None:
            project_name = config["project_name"]
        else:
            project_name = project_path

        if model_name is None or last_pt_path is None:
            (
                last_pt_path,
                model_name,
            ) = _get_current_model_name_and_weight_from_latest_run(config)

        train.run(
            cfg=config["model_cfg"],
            data=config["data_config"],
            hyp=config["model_hyp"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            project=project_name,
            name=model_name,
            resume=last_pt_path,
        )
    except FileNotFoundError as fnfe:
        print(fnfe)


def _get_current_model_name_and_weight_from_latest_run(
    config: dict,
) -> Tuple[Path, str]:
    wandb_project_dir = Path(config["project_name"])
    model_name = config["model_name"]
    if not wandb_project_dir.exists():
        raise FileNotFoundError(f"Directory [{wandb_project_dir}] doesn't exist!")

    current_idx = len([_dir for _dir in wandb_project_dir.iterdir() if _dir.is_dir()])
    assert current_idx != 0, f"Empty directory! [{wandb_project_dir}]"

    if current_idx == 1:
        latest_run_dir = wandb_project_dir / f"{model_name}"
        current_model_name = f"{model_name}"
    else:
        latest_run_dir = wandb_project_dir / f"{model_name}_{current_idx}"
        current_model_name = f"{model_name}_{current_idx}"

    if not latest_run_dir.exists():
        raise FileNotFoundError(
            f"Latest run directory could not be found. Actual path: [{latest_run_dir}]"
        )
    last_pt_path = Path(latest_run_dir, "weights/last.pt")

    if not last_pt_path.is_file():
        # TODO: recursively search for next available
        raise FileNotFoundError(
            f"last.pt file not found at: '{last_pt_path}'! "
            + "\nNot able to resume training."
        )

    return last_pt_path, current_model_name


if __name__ == "__main__":
    config_path = CONFIG.MODEL_TRAINING_CONFIG_PATH
    main(config_path)
