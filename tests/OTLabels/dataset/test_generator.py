from pathlib import Path

from dataset.generator import SampleType

from OTLabels.dataset.generator import ImageDirectory


class TestImageDirectory:
    def test_relative_to(self) -> None:
        base_path = Path("base_path")
        sample_type = SampleType.CORRECT_CLASSIFICATION
        resolution = "800x600"
        classification = "car"
        directory = ImageDirectory(
            base_path=base_path,
            sample_type=sample_type,
            resolution=resolution,
            classification=classification,
        )
        other_path = Path("other_path")
        expected_directory = ImageDirectory(
            base_path=other_path,
            sample_type=sample_type,
            resolution=resolution,
            classification=classification,
        )

        other_output = directory.relative_to(other_path)
        assert expected_directory.path == other_output
