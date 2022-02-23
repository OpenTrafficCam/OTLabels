from pathlib import Path
import pytest


@pytest.fixture
def test_resources_dir() -> Path:
    return Path(Path(__file__).parent, "resources")
