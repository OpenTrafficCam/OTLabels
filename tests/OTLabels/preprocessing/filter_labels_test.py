import shutil

import pytest
from pathlib import Path
import pandas as pd
from PIL import Image

from OTLabels.preprocessing.filter_labels import _is_bbox_to_img_area_ratio_in_thresh
from OTLabels.preprocessing.filter_labels import _get_bboxes
from OTLabels.preprocessing.filter_labels import _get_cvat_yolo_ann_path_from_img_path
from OTLabels.preprocessing.filter_labels import _filter_bboxes_with_bbox_img_ratio


@pytest.fixture
def img_size():
    return [640, 480]


@pytest.fixture
def neg_img_size():
    return [-640, -480]


@pytest.fixture
def test_dataset_dir(test_resources_dir):
    return Path(test_resources_dir, "example_dataset")


@pytest.fixture
def img_path_list(test_dataset_dir):
    return [
        Path(test_dataset_dir, "images/000000000139.jpg"),
        Path(test_dataset_dir, "images/000000000724.jpg"),
        Path(test_dataset_dir, "images/000000000785.jpg"),
        Path(test_dataset_dir, "images/000000000872.jpg"),
        Path(test_dataset_dir, "images/000000000885.jpg"),
    ]


@pytest.fixture
def ann_path_list(test_dataset_dir):
    return [
        Path(test_dataset_dir, "labels/000000000139.txt"),
        Path(test_dataset_dir, "labels/000000000724.txt"),
        Path(test_dataset_dir, "labels/000000000785.txt"),
        Path(test_dataset_dir, "labels/000000000872.txt"),
        Path(test_dataset_dir, "labels/000000000885.txt"),
    ]


@pytest.fixture
def df_anns(ann_path_list):
    dfs = [pd.read_csv(ann_path, header=None, sep=" ") for ann_path in ann_path_list]
    return dfs


@pytest.fixture
def example_yolov5_dir(test_resources_dir, img_path_list: list[Path]):
    """Creates an example YOLOv5 containing images and labels."""
    test_example_dir = Path(test_resources_dir, "test_example")
    img_dir = test_example_dir / "images"
    labels_dir = test_example_dir / "labels"
    img_dir.mkdir(exist_ok=True, parents=True)
    labels_dir.mkdir(exist_ok=True, parents=True)

    anns = [
        ["0 75 75 300 400", "4 12 17 10 10"],
        ["5 1 1 10 10", "21 4 4 1 1"],
    ]
    # Copy images to temp folder
    for img_path, ann in zip(img_path_list[0:2], anns):
        dst = Path(img_dir, f"{img_path.name}")
        shutil.copyfile(src=img_path, dst=dst)

        # Create ann_files
        ann_path = Path(test_example_dir, f"labels/{img_path.stem}.txt")
        ann_path.touch(exist_ok=True)
        with open(ann_path, "w") as f:
            for a in ann:
                f.write(a)
                f.write("\n")

    yield test_example_dir
    shutil.rmtree(test_example_dir)


@pytest.mark.parametrize(
    "bbox_size,thresh", [([40, 20], [0.3, 1]), ([400, 400], [0.3, 0.5])]
)
def test_is_bbox_to_img_area_ratio_in_thresh_ratioOutsideThresh_returnsFalse(
    img_size, bbox_size, thresh
):
    img_width, img_height = img_size
    bbox_width, bbox_height = bbox_size
    lower_thresh, upper_thresh = thresh
    assert not _is_bbox_to_img_area_ratio_in_thresh(
        bbox_width=bbox_width,
        bbox_height=bbox_height,
        img_width=img_width,
        img_height=img_height,
        lower_thresh=lower_thresh,
        upper_thresh=upper_thresh,
    )


@pytest.mark.parametrize(
    "bbox_size,thresh",
    [([0, 0], [0, 1]), ([640, 480], [0, 1]), ([324, 171], [0.15, 0.5])],
)
def test_is_bbox_to_img_area_ratio_in_thresh_ratioInsideThresh_returnsTrue(
    img_size, bbox_size, thresh
):
    img_width, img_height = img_size
    bbox_width, bbox_height = bbox_size
    lower_thresh, upper_thresh = thresh
    assert _is_bbox_to_img_area_ratio_in_thresh(
        bbox_width=bbox_width,
        bbox_height=bbox_height,
        img_width=img_width,
        img_height=img_height,
        lower_thresh=lower_thresh,
        upper_thresh=upper_thresh,
    )


def test_is_bbox_to_img_area_ratio_in_thresh_negImgSizeValues_RaiseValueError(
    neg_img_size,
):
    img_width, img_height = neg_img_size

    with pytest.raises(ValueError) as ve:
        _is_bbox_to_img_area_ratio_in_thresh(
            bbox_width=400,
            bbox_height=400,
            img_width=img_width,
            img_height=img_height,
            lower_thresh=0.3,
            upper_thresh=1,
        )
    assert "Image width and height must be positive values." in str(ve.value)


def test_is_bbox_to_img_area_ratio_in_thresh_negBboxValues_RaiseValueError(img_size):
    img_width, img_height = img_size

    with pytest.raises(ValueError) as ve:
        _is_bbox_to_img_area_ratio_in_thresh(
            bbox_width=-400,
            bbox_height=-400,
            img_width=img_width,
            img_height=img_height,
            lower_thresh=0.3,
            upper_thresh=1,
        )
    assert "Bbox width and height must be positive values." in str(ve.value)


def test_is_bbox_to_img_area_ratio_in_thresh_bboxValuesGtImg_RaiseAssertionError(
    img_size,
):
    img_width, img_height = img_size

    with pytest.raises(AssertionError):
        _is_bbox_to_img_area_ratio_in_thresh(
            bbox_width=800,
            bbox_height=800,
            img_width=img_width,
            img_height=img_height,
            lower_thresh=0,
            upper_thresh=1,
        )


def test_get_bboxes_validAnnPath_returnsCorrectBBoxes(ann_path_list):
    ann_path = ann_path_list[0]
    bboxes = _get_bboxes(ann_path)
    b1_cls, b1_x, b1_y, b1_w, b1_h = bboxes[0]
    b2_cls, b2_x, b2_y, b2_w, b2_h = bboxes[1]

    assert len(bboxes) == 2 and len(bboxes[0]) == 5 and len(bboxes[1]) == 5
    assert (
        b1_cls == 0
        and b1_x == 0.686445
        and b1_y == 0.53196
        and b1_w == 0.082891
        and b1_h == 0.323967
    )
    assert (
        b2_cls == 0
        and b2_x == 0.612484
        and b2_y == 0.446197
        and b2_w == 0.023625
        and b2_h == 0.083897
    )


def test_get_cvat_yolo_ann_path_from_img_path_withExistingLabels_returnsTrue(
    img_path_list,
):
    for img_path in img_path_list:
        ann_path = _get_cvat_yolo_ann_path_from_img_path(img_path)
        assert ann_path.is_file()


def test_get_cvat_yolo_ann_path_from_img_path_withNotExistingLabels_returnsFalse():
    example_img_path = Path("path/to/dir/images/frame_1.jpeg")
    ann_path = _get_cvat_yolo_ann_path_from_img_path(example_img_path)
    assert not ann_path.is_file()
    assert ann_path == Path("path/to/dir/labels/frame_1.txt")


def test_filter_bboxes_with_bbox_img_ratio_noThreshApplied_normalized_returnsSameDets(
    df_anns,
):
    for df_ann in df_anns:
        filtered_dets = _filter_bboxes_with_bbox_img_ratio(
            anns=df_ann, img_height=1, img_width=1, lower_thresh=0, upper_thresh=1
        )
        assert len(filtered_dets) == len(df_ann.index)


def test_filter_bboxes_with_bbox_img_ratio_threshDiscardAll_normalized_returnsNoDets(
    img_path_list,
    df_anns,
):
    for df_ann in df_anns:
        filtered_dets = _filter_bboxes_with_bbox_img_ratio(
            anns=df_ann,
            img_width=1,
            img_height=1,
            lower_thresh=0,
            upper_thresh=0,
        )
        for img_bboxes in filtered_dets:
            assert len(img_bboxes) == 0


def test_filter_bboxes_with_bbox_img_ration_normalizedFalseAsParam_returnDets(
    example_yolov5_dir,
):
    img_paths = [
        img_path
        for img_path in Path(example_yolov5_dir, "images").iterdir()
        if img_path.is_file()
    ]
    df_anns = [
        pd.read_csv(ann_path, header=None, sep=" ")
        for ann_path in Path(example_yolov5_dir, "labels").iterdir()
        if ann_path.is_file()
    ]

    filtered_bboxes = []
    for img_path, df_ann in zip(img_paths, df_anns):
        img_width, img_height = Image.open(img_path).size

        filtered_bbox_of_img = _filter_bboxes_with_bbox_img_ratio(
            anns=df_ann,
            img_width=img_width,
            img_height=img_height,
            lower_thresh=0.1,
            upper_thresh=1,
        )
        filtered_bboxes.append(filtered_bbox_of_img)

    assert len(filtered_bboxes[0]) == 1
    assert len(filtered_bboxes[1]) == 0
