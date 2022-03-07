import shutil

import pytest
from pathlib import Path

from OTLabels.helpers.files import get_yolov5_img_path_from_ann_path
from OTLabels.helpers.files import write_yolov5_anns_to_file
from OTLabels.helpers.files import read_cvat_labels_file


@pytest.fixture
def empty_dir(test_resources_dir: Path) -> Path:
    empty_dir = test_resources_dir / "empty"
    empty_dir.mkdir()
    yield empty_dir

    shutil.rmtree(empty_dir)


@pytest.fixture
def anns() -> list:
    anns = [[1, 0.13, 0.14, 0.2, 0.2], [4, 0.45, 0.3, 0.01, 0.04]]
    return anns


@pytest.mark.parametrize("img_ext", [".jpg", "jpg", "JPG"])
def test_get_yolov5_img_path_from_ann_path(img_ext):
    ann_path = Path("path/to/yolo_dir/labels/frame_1.txt")
    img_path = get_yolov5_img_path_from_ann_path(ann_path, img_ext)
    if img_ext.startswith("."):
        assert img_path == Path(f"path/to/yolo_dir/images/frame_1{img_ext}")
    else:
        assert img_path == Path(f"path/to/yolo_dir/images/frame_1.{img_ext}")


def test_write_yolov5_anns_to_file(empty_dir: Path, anns: list):
    dest_path = empty_dir / "frame_1.txt"
    write_yolov5_anns_to_file(anns, dest_path)
    assert dest_path.is_file()

    with open(dest_path, "r") as f:
        read_anns = f.readlines()

    expected_read_anns_0 = "1 0.13 0.14 0.2 0.2\n"
    expected_read_anns_1 = "4 0.45 0.3 0.01 0.04\n"
    assert read_anns[0] == expected_read_anns_0
    assert read_anns[1] == expected_read_anns_1


def test_read_cvat_labels_file(test_resources_dir: Path):
    cvat_labels_path = test_resources_dir / "misc/labels_CVAT.txt"
    label_data = read_cvat_labels_file(cvat_labels_path, delimiter=",")
    assert label_data["person"] == 0
    assert label_data["bicycle"] == 1
    assert label_data["car"] == 2
    assert label_data["motorcycle"] == 3
    assert label_data["bus"] == 5
    assert label_data["truck"] == 7
