import pytest
from pathlib import Path

from OTLabels.preprocessing.filter_labels import _is_bbox_to_img_area_ratio_in_thresh
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
