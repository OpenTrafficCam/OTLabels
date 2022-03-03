import shutil

from pathlib import Path
import pytest


@pytest.fixture
def test_resources_dir() -> Path:
    return Path(Path(__file__).parent, "resources")


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
