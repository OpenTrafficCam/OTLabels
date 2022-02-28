import pytest
from pathlib import Path

from resume import _get_curr_model_weight_and_name_from_latest_run


@pytest.mark.parametrize("num_run", [1, 2, 3])
def test_get_curr_model_weight_and_name_from_latest_run(wandb_project_dir, num_run):
    config = wandb_project_dir(num_run)
    project_dir = config["project_name"]
    model_name = config["model_name"]
    last_pt_rel = "weights/last.pt"

    curr_last_pt, curr_model_name = _get_curr_model_weight_and_name_from_latest_run(
        config
    )
    if num_run == 1:
        assert curr_last_pt == Path(project_dir, f"{model_name}", last_pt_rel)
        assert curr_model_name == model_name
    else:
        assert curr_last_pt == Path(project_dir, f"{model_name}_{num_run}", last_pt_rel)
        assert curr_model_name == f"{model_name}_{num_run}"


def test_get_curr_model_weight_and_name_from_latest_run_emptyDir_raiseFileNotFoundError(
    wandb_project_dir,
):
    config = wandb_project_dir(0)
    with pytest.raises(FileNotFoundError):
        _, _ = _get_curr_model_weight_and_name_from_latest_run(config)


def test_get_curr_model_weight_and_name_from_latest_run_NoLastPt_raiseFileNotFoundError(
    wandb_project_dir,
):
    config = wandb_project_dir(1)
    Path(config["project_name"], config["model_name"], "weights/last.pt").unlink()
    with pytest.raises(FileNotFoundError):
        _, _ = _get_curr_model_weight_and_name_from_latest_run(config)
