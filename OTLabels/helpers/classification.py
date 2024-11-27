import json
from os import PathLike


def load_classes(class_file: str | PathLike) -> dict[str, int]:
    with open(class_file) as json_file:
        return json.load(json_file)
    return {}
