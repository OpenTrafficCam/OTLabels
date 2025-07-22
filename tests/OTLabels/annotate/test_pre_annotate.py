from pathlib import Path

import pytest

from OTLabels.annotate.otc_classes import OtcClass
from OTLabels.annotate.pre_annotate import (
    Image,
    PreAnnotateImages,
    drop_images_too_close,
    reorder_samples,
)
from OTLabels.dataset.generator import ImageDirectory, SampleType

DEFAULT_IMAGE_DIRECTORY = ImageDirectory(
    base_path=Path(""),
    sample_type=SampleType.CORRECT_CLASSIFICATION,
    resolution="",
    classification=OtcClass.CAR,
)


class TestYOLOv8:
    def test_init_config_path(self) -> None:
        expected_result = "data/image_data/Aachen_OTCamera12/images"
        calculator = PreAnnotateImages(
            config_file="OTLabels/config/training_data.json",
            class_file="OTLabels/config/labels_COCO.json",
            filter_sites=["Aachen_OTCamera12"],
        )
        result: PreAnnotateImages = calculator.config["Aachen_OTCamera12"]["image_path"]
        assert expected_result == result

    @pytest.mark.parametrize(
        "label_class, expected_result",
        [
            ("person", 0),
            ("bicycle", 1),
            ("car", 2),
            ("motorcycle", 3),
            ("bus", 5),
            ("truck", 7),
        ],
    )
    def test_init_labels(self, label_class, expected_result) -> None:
        calculator = PreAnnotateImages(
            config_file="OTLabels/config/training_data.json",
            class_file="OTLabels/config/labels_COCO.json",
        )
        result: PreAnnotateImages = calculator.classes[label_class]
        assert expected_result == result


class Site:

    def __init__(self, name: str) -> None:
        self.image_0 = create_image(0, name)
        self.image_1 = create_image(1, name)
        self.image_2 = create_image(2, name)
        self.image_3 = create_image(3, name)
        self.image_4 = create_image(4, name)
        self.image_5 = create_image(5, name)
        self.image_6 = create_image(6, name)


def create_image(frame: int, site: str, directory=DEFAULT_IMAGE_DIRECTORY):
    return Image(
        image_directory=directory,
        image_path=directory.path / f"{site}_image_{frame}.png",
        site=site,
        cam_type="Standard",
        frame=frame,
    )


SITE_0 = Site("site-0")
SITE_1 = Site("site-1")


class TestImage:
    def test_move(self, test_data_tmp_dir: Path) -> None:
        directory = ImageDirectory(
            base_path=test_data_tmp_dir / "base",
            sample_type=SampleType.CORRECT_CLASSIFICATION,
            resolution="720x480",
            classification=OtcClass.CAR,
        )
        image = create_image(directory=directory, site="site-0", frame=0)
        image.image_path.parent.mkdir(parents=True, exist_ok=True)
        image.image_path.touch()

        assert image.image_path.exists()
        output_folder = test_data_tmp_dir / "output"
        image.move(to=output_folder)
        expected_output = directory.relative_to(output_folder) / image.image_path.name
        assert expected_output.exists()


@pytest.mark.parametrize(
    "input_images, expected_images, frame_gap",
    [
        (
            [SITE_0.image_0, SITE_0.image_1, SITE_0.image_2, SITE_0.image_3],
            [SITE_0.image_0, SITE_0.image_2],
            1,
        ),
        (
            [
                SITE_0.image_0,
                SITE_0.image_1,
                SITE_0.image_2,
                SITE_0.image_3,
                SITE_0.image_4,
                SITE_0.image_5,
                SITE_0.image_6,
            ],
            [SITE_0.image_0, SITE_0.image_3, SITE_0.image_6],
            2,
        ),
        (
            [SITE_0.image_0, SITE_1.image_1],
            [SITE_0.image_0, SITE_1.image_1],
            1,
        ),
        (
            [
                SITE_0.image_0,
                SITE_0.image_1,
                SITE_0.image_2,
                SITE_1.image_1,
                SITE_1.image_2,
                SITE_1.image_3,
            ],
            [SITE_0.image_0, SITE_0.image_2, SITE_1.image_1, SITE_1.image_3],
            1,
        ),
    ],
)
def test_drop_images_too_close(
    input_images: list[Image],
    expected_images: list[Image],
    frame_gap: int,
) -> None:
    assert drop_images_too_close(input_images, frame_gap=frame_gap) == expected_images


@pytest.mark.parametrize(
    "input_images, expected_images",
    [
        (
            [SITE_0.image_0, SITE_0.image_1, SITE_0.image_2],
            [SITE_0.image_0, SITE_0.image_1, SITE_0.image_2],
        ),
        (
            [SITE_0.image_0, SITE_1.image_1],
            [SITE_0.image_0, SITE_1.image_1],
        ),
        (
            [
                SITE_0.image_0,
                SITE_0.image_1,
                SITE_0.image_2,
                SITE_1.image_2,
                SITE_1.image_3,
                SITE_1.image_4,
            ],
            [
                SITE_0.image_0,
                SITE_1.image_2,
                SITE_0.image_1,
                SITE_1.image_3,
                SITE_0.image_2,
                SITE_1.image_4,
            ],
        ),
    ],
)
def test_reorder_samples(
    input_images: list[Image], expected_images: list[Image]
) -> None:
    assert reorder_samples(input_images) == expected_images
