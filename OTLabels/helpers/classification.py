import json
from os import PathLike

from annotate.otc_classes import OtcClass


def load_classes(class_file: str | PathLike) -> dict[OtcClass, int]:
    with open(class_file) as json_file:
        return {OtcClass.from_string(key): value for key, value in json.load(json_file)}
    return {}
