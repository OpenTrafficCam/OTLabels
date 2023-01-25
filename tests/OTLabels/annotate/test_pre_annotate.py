import pytest

from OTLabels.annotate.pre_annotate import PreAnnotateImages


class TestYOLOv8:
    def test_init_config_path(self) -> None:
        erwartetes_ergebnis = "data/image_data/Aachen_OTCamera12/images"
        rechner = PreAnnotateImages(
            config_file="OTLabels/config/training_data.json",
            class_file="OTLabels/config/labels_COCO.json",
            filter_sites=["Aachen_OTCamera12"],
        )
        ergebnis: PreAnnotateImages = rechner.config["Aachen_OTCamera12"]["image_path"]
        assert erwartetes_ergebnis == ergebnis

    @pytest.mark.parametrize(
        "label_class, erwartetes_ergebnis",
        [
            ("person", 0),
            ("bicycle", 1),
            ("car", 2),
            ("motorcycle", 3),
            ("bus", 5),
            ("truck", 7),
        ],
    )
    def test_init_labels(self, label_class, erwartetes_ergebnis) -> None:
        rechner = PreAnnotateImages(
            config_file="OTLabels/config/training_data.json",
            class_file="OTLabels/config/labels_COCO.json",
        )
        ergebnis: PreAnnotateImages = rechner.classes[label_class]
        assert erwartetes_ergebnis == ergebnis
