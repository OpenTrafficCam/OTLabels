from typing import Tuple

from yolov5 import train
from pathlib import Path

import config as CONFIG
from config import load_config


def main(config_path):
    config = load_config(config_path)
    try:
        # Use last.pt as model weight and create another training session
        last_pt, current_model_name = _get_last_pt_path_and_next_model_name(config)
        train.run(
            weights=last_pt,
            cfg=config["model_cfg"],
            data=config["data_config"],
            hyp=config["model_hyp"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            project=config["project_name"],
            name=current_model_name,
            resume=True,
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
            resume=True,
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

    current_model_name = f"{model_name}_{current_idx}"
    assert last_pt_path.is_file(), f"File '{last_pt_path}' doesn't exist!"

    return last_pt_path, current_model_name


if __name__ == "__main__":
    config_path = CONFIG.MODEL_TRAINING_CONFIG_PATH
    main(config_path)
