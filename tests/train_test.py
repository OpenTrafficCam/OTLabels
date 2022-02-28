import shutil

import pytest
from pathlib import Path

from train import _get_last_pt_and_next_model_name
from train import LastPtNotFoundError
from train import _search_last_pt_recursively


@pytest.fixture
def wandb_project_dir(test_resources_dir: Path) -> Path:
    example = test_resources_dir / "example"

    def create_wandb_project_dir(num_runs):
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
    shutil.rmtree(test_resources_dir / "path")


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


def test_search_last_pt_recursively_lastTwoLastPtRemoved_returnsCorrectLastPt(
    wandb_project_dir,
):
    idx = 3
    config = wandb_project_dir(3)
    project_dir = config["project_name"]
    model_name = config["model_name"]
    Path(project_dir, f"{model_name}_2/weights/last.pt").unlink()
    Path(project_dir, f"{model_name}_3/weights/last.pt").unlink()

    last_pt_path = _search_last_pt_recursively(project_dir, model_name, idx)

    last_pt_run_1_path = Path(project_dir, f"{model_name}/weights/last.pt")
    assert last_pt_path == last_pt_run_1_path


def test_search_last_pt_recursively_noMissingLastPt_returnsCorrectLastPt(
    wandb_project_dir,
):
    idx = 3
    config = wandb_project_dir(3)
    project_dir = config["project_name"]
    model_name = config["model_name"]
    last_pt_path = _search_last_pt_recursively(project_dir, model_name, idx)

    last_pt_run_1_path = Path(project_dir, f"{model_name}_{idx}/weights/last.pt")
    assert last_pt_path == last_pt_run_1_path


@pytest.mark.parametrize("num_run", [1, 2, 3])
def test_search_last_pt_recursively_noLastPtExists_raiseFileNotFoundError(
    wandb_project_dir, num_run
):
    last_pt_rel = "weights/last.pt"

    config = wandb_project_dir(num_run)
    project_dir = config["project_name"]
    model_name = config["model_name"]

    for j in range(1, num_run + 1):
        # delete all last.pt files
        if j == 1:
            Path(project_dir, f"{model_name}", last_pt_rel).unlink()
        else:
            Path(project_dir, f"{model_name}_{j}", last_pt_rel).unlink()
    with pytest.raises(FileNotFoundError):
        _search_last_pt_recursively(project_dir, model_name, num_run)
