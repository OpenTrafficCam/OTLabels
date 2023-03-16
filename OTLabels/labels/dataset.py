"""Create and administrate FiftyOne datasets"""

import fiftyone


class Dataset:
    config_file: str
    name: str
    filter_sites: list = []
    permanent: bool = False

    def create(self) -> fiftyone.Dataset:
        pass
