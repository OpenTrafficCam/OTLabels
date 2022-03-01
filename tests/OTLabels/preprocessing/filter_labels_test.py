import pytest
from pathlib import Path

from OTLabels.preprocessing.filter_labels import _bbox_to_img_area_ratio_lt_thresh
from OTLabels.preprocessing.filter_labels import _get_bboxes


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


def test_bbox_to_img_area_ratio_lt_thresh_ratioBelowThreshold_returnsTrue(img_size):
    img_width, img_height = img_size
    assert _bbox_to_img_area_ratio_lt_thresh(
        bbox_width=40,
        bbox_height=20,
        img_width=img_width,
        img_height=img_height,
        thresh=0.3,
    )


def test_bbox_to_img_area_ratio_lt_thresh_ratioAboveThreshold_returnsFalse(img_size):
    img_width, img_height = img_size
    assert not _bbox_to_img_area_ratio_lt_thresh(
        bbox_width=400,
        bbox_height=400,
        img_width=img_width,
        img_height=img_height,
        thresh=0.3,
    )


def test_bbox_to_img_area_ratio_lt_thresh_negImgSizeValues_RaiseValueError(
    neg_img_size,
):
    img_width, img_height = neg_img_size

    with pytest.raises(ValueError):
        _bbox_to_img_area_ratio_lt_thresh(
            bbox_width=400,
            bbox_height=400,
            img_width=img_width,
            img_height=img_height,
            thresh=0.3,
        )


def test_bbox_to_img_area_ratio_lt_thresh_negBboxValues_RaiseValueError(img_size):
    img_width, img_height = img_size

    with pytest.raises(ValueError):
        _bbox_to_img_area_ratio_lt_thresh(
            bbox_width=-400,
            bbox_height=-400,
            img_width=img_width,
            img_height=img_height,
            thresh=0.3,
        )


def test_bbox_to_img_area_ratio_lt_thresh_bboxValuesGtImg_RaiseAssertionError(img_size):
    img_width, img_height = img_size

    with pytest.raises(AssertionError):
        _bbox_to_img_area_ratio_lt_thresh(
            bbox_width=800,
            bbox_height=800,
            img_width=img_width,
            img_height=img_height,
            thresh=0.3,
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
