import csv
from typing import Union

from pathlib import Path


def get_yolov5_img_path_from_ann_path(label_path: Path, img_ext: str):
    if img_ext.startswith("."):
        img_ext = img_ext[1:]
    return Path(label_path.parent.with_stem("images"), f"{label_path.stem}.{img_ext}")


def write_yolov5_anns_to_file(anns: list, dest: Path):
    with open(dest, "w") as f:
        for ann in anns:
            assert len(ann) == 5, (
                "A YOLOv5 annotation should consist of 5 elements(x,y,w,h).\n"
                + f"Actual number of elements: {len(ann)}"
            )
            _cls, x, y, w, h = ann
            f.write(f"{_cls} {x} {y} {w} {h}")
            f.write("\n")


def read_cvat_labels_file(file_path: Union[str, Path], delimiter: str) -> dict:
    """
    Reads from a csv file and returns a dictionary without the header.

    Returns:
        A dictionary, where the class_names are keys and the CatIds are integers.
    """
    data = {}
    with open(file_path, "r") as f:
        csv_reader = csv.reader(f, delimiter=delimiter)
        for idx, row in enumerate(csv_reader):
            if idx != 0:
                class_name, cat_id = row
                data.update({class_name: int(cat_id)})

    return data
