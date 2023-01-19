"""Create and administrate FiftyOne datasets"""

# Import fiftyone
import fiftyone as fo


class dataset:

    config_file: str
    name: str
    filter_sites: list = []
    permanent: bool = False

    def create(self) -> fo.Dataset:
        pass
