import pytest

from OTLabels.preprocessing.filter_labels import _bbox_to_img_area_ratio_lt_thresh


@pytest.fixture
def img_size():
    return [640, 480]


@pytest.fixture
def neg_img_size():
    return [-640, -480]


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
