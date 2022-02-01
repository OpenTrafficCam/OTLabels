# Copyright (C) 2021 OpenTrafficCam Contributors
# <https://github.com/OpenTrafficCam
# <team@opentrafficcam.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# TODO: docstrings in cvat_to_coco

from pathlib import Path
import shutil
from pycocotools.coco import COCO
import pandas as pd
import os


def _get_coco_cats(cat_file, ann_file):
    cats = pd.read_csv(cat_file)["Cat"].tolist()
    coco = COCO(ann_file)
    cat_ids = coco.getCatIds(catNms=cats)
    labels_coco = pd.DataFrame(list(zip(cats, cat_ids)), columns=["Cat", "CatId"])
    return labels_coco


def _file_list_cvat(file, suffix):
    file = Path(file)
    dir = file.with_suffix("")
    files = dir.glob(f"obj_train_data/*.{suffix}")
    return [str(file) for file in files]


def _get_file_list(file, suffix):
    file = Path(file)
    dir = file.with_suffix("")
    files = dir.glob("*.{}".format(suffix))
    return [str(file) for file in files]


def _unzip(file):
    file = Path(file)
    dir = file.with_suffix("")
    shutil.unpack_archive(file, dir)
    img_files = _file_list_cvat(file, "png")
    ann_files = _file_list_cvat(file, "txt")
    return img_files, ann_files, dir


def _copy_files(src_files, dest_path, counter):
    if not Path(dest_path).exists():
        Path(dest_path).mkdir()
    for f in src_files:
        new_name = f"{counter}_{Path(f).name}"
        shutil.copy(f, Path(dest_path, new_name))


def _create_label_dict(labels_cvat, labels_yolo):
    labels = pd.merge(
        labels_cvat, labels_yolo, how="inner", on="Cat", suffixes=["_CVAT", "_COCO"]
    )
    label_dict = {}
    for i in labels["Cat"]:
        label_dict.update(
            {
                labels.loc[labels["Cat"] == i, "CatId_CVAT"]
                .values[0]: labels.loc[labels["Cat"] == i, "CatId_COCO"]
                .values[0]
                - 1
            }
        )
    return label_dict


def _copy_files_convert(ann_files, ann_path, labels_cvat, labels_yolo, counter):
    if not Path(ann_path).exists():
        Path(ann_path).mkdir()

    label_dict = _create_label_dict(labels_cvat, labels_yolo)

    for f in ann_files:
        file_name = Path(f).name
        if os.stat(f).st_size > 0:
            file_labels = pd.read_csv(f, header=None, sep=" ")
        else:
            shutil.copy(f, Path(ann_path, f"{counter}_{file_name}"))

        file_labels[0] = file_labels[0].map(label_dict)
        file_labels = file_labels.dropna()
        file_labels[0] = file_labels[0].astype(int)
        file_labels.to_csv(
            ann_path + "/" + str(counter) + "_" + file_name,
            header=False,
            sep=" ",
            index=False,
            line_terminator="\n",
        )


def _file_structure(cvat_file, dest_path, labels_cvat, labels_yolo, name, counter):
    img_files, ann_files, src_path = _unzip(cvat_file)
    img_path = Path(dest_path, f"images/{name}")
    ann_path = Path(dest_path, f"labels/{name}")

    _copy_files(img_files, img_path, counter)

    _copy_files_convert(ann_files, ann_path, labels_cvat, labels_yolo, counter)
    shutil.rmtree(src_path)
    return ann_files


def main(dest_path, cvat_dir, labels_cvat_path, coco_ann_file_path, name):
    labels_cvat = pd.read_csv(labels_cvat_path)
    labels_yolo = _get_coco_cats(labels_cvat_path, coco_ann_file_path)
    assert len(labels_cvat) == len(labels_yolo), "CVAT Labels and YOLO Labels differ!"

    n = 0
    if Path(cvat_dir).is_file():
        _file_structure(cvat_dir, dest_path, labels_cvat, labels_yolo, name, n)
    elif Path(cvat_dir).is_dir:
        cvat_files = _get_file_list(cvat_dir, "zip")
        for file in cvat_files:
            _file_structure(file, dest_path, labels_cvat, labels_yolo, name, n)
            n = n + 1


if __name__ == "__main__":
    """dest_path = "D:/OTC/OTLabels/OTLabels/data/coco"
    coco_ann_file = (
        "D:/OTC/OTLabels/OTLabels/data/coco/annotations/instances_val2017.json"
    )
    cat_file = "D:/OTC/OTLabels/OTLabels/labels_CVAT.txt"
    cvat_dir = "C:/Users/MichaelHeilig/Downloads/Radeberg"
    name = "radeberger-00"

    labels_cvat = pd.read_csv(cat_file)
    labels_yolo = _get_coco_cats(cat_file, coco_ann_file)

    main(dest_path, cvat_dir, labels_cvat_path, labels_yolo, name)
    """
    pass
