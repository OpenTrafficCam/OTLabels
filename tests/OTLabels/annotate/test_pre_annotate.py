import pytest

from OTLabels.annotate.pre_annotate import PreAnnotateImages


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
