import shutil

import pytest
from pathlib import Path

from train import _get_last_pt_path_and_next_model_name


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

        return project_name, model_name

    yield create_wandb_project_dir

    shutil.rmtree(example)


@pytest.mark.parametrize("num_runs", [1, 2, 3])
def test_get_last_pt_path_and_next_model_name_returnsCorrectPathModelname(
    wandb_project_dir, num_runs
):
    project_name, model_name = wandb_project_dir(num_runs)
    config = {}
    config["project_name"] = project_name
    config["model_name"] = model_name
    result_last_pt, result_next_model_name = _get_last_pt_path_and_next_model_name(
        config
    )

    if num_runs == 1:
        correct = Path(project_name, f"{model_name}/weights/last.pt")

    else:
        correct = Path(project_name, f"{model_name}_{num_runs}/weights/last.pt")

    assert result_last_pt == correct
    assert result_next_model_name == f"{model_name}_{num_runs + 1}"
