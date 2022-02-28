import shutil

import pytest
from pathlib import Path

from train import _get_last_pt_and_next_model_name
from train import LastPtNotFoundError


@pytest.fixture
def wandb_project_dir(test_resources_dir: Path) -> Path:
    example = test_resources_dir / "example"

    def create_wandb_project_dir(num_runs):
        assert num_runs >= 0, "Value needs to be non negative integer"

        project_name = example / f"data/runs/test_num_runs_{num_runs}"
        model_name = "yolov5-s_COCO"
        project_name.mkdir(parents=True, exist_ok=True)

        for i in range(1, num_runs + 1):
            if i <= 1:
                run_dir = project_name / f"{model_name}"
            else:
                run_dir = project_name / f"{model_name}_{i}"

            run_weights_dir = run_dir / "weights"
            run_best_pt = run_weights_dir / "best.pt"
            run_last_pt = run_weights_dir / "last.pt"

            run_dir.mkdir(parents=True, exist_ok=True)
            run_weights_dir.mkdir(parents=True, exist_ok=True)
            run_best_pt.touch(exist_ok=True)
            run_last_pt.touch(exist_ok=True)

        config = {}
        config["project_name"] = project_name
        config["model_name"] = model_name

        return config

    yield create_wandb_project_dir

    shutil.rmtree(example)


@pytest.fixture
def empty_dir_config(test_resources_dir: Path) -> dict:
    config = {}
    config["project_name"] = Path(test_resources_dir, "path/to/dir")
    config["model_name"] = "model_name"
    config["project_name"].mkdir(parents=True, exist_ok=True)
    yield config
    shutil.rmtree(config["project_name"])


@pytest.mark.parametrize("num_runs", [1, 2, 3])
def test_get_last_pt_and_next_model_name_returnsCorrectPathModelname(
    wandb_project_dir, num_runs
):
    config = wandb_project_dir(num_runs)
    project_name = config["project_name"]
    model_name = config["model_name"]

    result_last_pt, result_next_model_name = _get_last_pt_and_next_model_name(config)

    if num_runs == 1:
        correct = Path(project_name, f"{model_name}/weights/last.pt")

    else:
        correct = Path(project_name, f"{model_name}_{num_runs}/weights/last.pt")

    assert result_last_pt == correct
    assert result_next_model_name == f"{model_name}_{num_runs + 1}"


def test_get_last_pt_and_next_model_name_noLastPtAsParam_raiseLastPtNotFoundError(
    empty_dir_config,
):
    project_name = empty_dir_config["project_name"]
    model_name = empty_dir_config["model_name"]
    Path(project_name, model_name, "weights").mkdir(parents=True, exist_ok=True)
    with pytest.raises(LastPtNotFoundError):
        _get_last_pt_and_next_model_name(empty_dir_config)


def test_get_last_pt_and_next_model_name_emptDirAsParam_raiseFileNotFoundError(
    empty_dir_config,
):
    with pytest.raises(FileNotFoundError):
        _get_last_pt_and_next_model_name(empty_dir_config)


def test_get_last_pt_and_next_model_name_noExistingDirAsPam_raiseFileNotFoundError():
    config = {}
    config["project_name"] = Path("path/to/dir/not/exist")
    config["model_name"] = "model_name"

    with pytest.raises(FileNotFoundError):
        _get_last_pt_and_next_model_name(config)
