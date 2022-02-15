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

# TODO: docstrings in filter_labels

from pathlib import Path
import pandas as pd
import shutil
import random
from tqdm import tqdm


def _reset_labels(labels):
    label_dict = {}
    row = 0
    for i in labels["Cat"]:
        label_dict.update({labels.loc[labels["Cat"] == i, "CatId"].values[0]: row})
        row = row + 1
    return label_dict


def _file_list(file, suffix):
    file = Path(file)
    print('Reading "{file}\\*{suffix}" files...'.format(file=file, suffix=suffix))
    dir = file.with_suffix("")
    if suffix == "":
        files = dir.glob("*")
    else:
        files = dir.glob("*.{}".format(suffix))
    return [str(file) for file in files]


def _has_files(dir_path):
    directory = Path(dir_path)

    if directory.is_dir():
        raise NotADirectoryError(directory)

    for f in directory.glob("**/*"):
        if f.is_file():
            return True

    return False


def _has_data(file_path):
    f = Path(file_path)
    return f.stat().st_size > 0


def _filter_labels(
    labels_filter,
    path,
    name,
    appendix,
    num_background,
    force_filtering,
    sample=1.0,
    reset_label_ids=False,
):
    source_dir = Path(path, f"labels/{name}")
    image_dir = Path(path, f"images/{name}")
    dest_dir_labels = Path(path, f"labels/{name}_filtered_{appendix}")
    dest_dir_imgs = Path(path, f"images/{name}_filtered_{appendix}")
    dest_dir_imgs_relative = f"./images/{name}_filtered_{appendix}"

    img_type = Path(_file_list(image_dir, "")[0]).suffix.lower()

    if not force_filtering and dest_dir_labels.exists() and dest_dir_imgs.exists():
        print(
            "Filtered data already exists. To force filtering set force_filtering FLAG "
            + f'to True or delete these folders: "{dest_dir_labels}", "{dest_dir_imgs}"'
        )
        return

    # Remove existing filtered data
    if Path(dest_dir_labels).exists():
        shutil.rmtree(dest_dir_labels)
    if Path(dest_dir_imgs).exists():
        shutil.rmtree(dest_dir_imgs)

    Path(dest_dir_labels).mkdir(parents=True)
    Path(dest_dir_imgs).mkdir(parents=True)

    labels = pd.read_csv(labels_filter)
    ann_files = _file_list(source_dir, "txt")

    if reset_label_ids:
        label_dict = _reset_labels(labels)

    print(
        f"Filter files in {source_dir} by labels "
        + ", ".join(str(e) for e in labels["Cat"].tolist())
        + "..."
    )

    rel_image_path_list = []
    image_list_source = []
    n = 0
    for f in tqdm(ann_files):

        write = random.uniform(0, 1) < sample

        file_name = Path(f).name
        image_name = Path(file_name).stem + img_type
        if _has_data(f):
            file_labels = pd.read_csv(f, header=None, sep=" ")
        else:
            continue

        file_labels = file_labels[file_labels[0].isin(labels["CatId"])]

        if len(file_labels) > 0:
            if reset_label_ids:
                file_labels[0] = file_labels[0].map(label_dict)
                file_labels = file_labels.dropna()
                file_labels[0] = file_labels[0].astype(int)
            if write:
                file_labels.to_csv(
                    Path(dest_dir_labels, file_name),
                    header=False,
                    sep=" ",
                    index=False,
                    line_terminator="\n",
                )
                rel_image_path_list.append(
                    "./" + str(Path(dest_dir_imgs_relative, image_name))
                )
                image_list_source.append(str(Path(image_dir, image_name)))
        else:
            if n < num_background:
                open(Path(dest_dir_labels, file_name), "a").close()  # NOTE: Reason?
                rel_image_path_list.append(
                    f"./{Path(dest_dir_imgs_relative, image_name)}"
                )
                image_list_source.append(str(Path(image_dir, image_name)))
            n = n + 1
            continue

    file_filtered_labels = Path(path, f"{name}_filtered_{appendix}.txt")
    print(f"Writing file with filtered labels to {file_filtered_labels} ...")

    with open(file_filtered_labels, "w") as f:
        f.write("\n".join(rel_image_path_list))

    print(f"Copying {len(rel_image_path_list)} images to {dest_dir_imgs} ...")
    for img in tqdm(image_list_source):
        shutil.copy(img, dest_dir_imgs)

    print("Done!")


def _get_bboxes(label_path):
    """
    Reads and parses the bounding boxes from a YOLOv5 annotation text file.

    The text file should contain bboxes in the form of `class x y w h\\n` where each
    value is separated by a whitespace and a newline marks a new bounding box.

    Args:
        label_path(str|PosixPath): Path to text file in YOLOv5 annotation format.

    Returns:
        A list containing the bounding box values in the form of [x, y, w, h].
    """
    with open(label_path, "r", errors="ignore") as f:
        lines = [line.rstrip("\n") for line in f.readlines()]
        bboxes = []
        for line in lines:
            vals = line.split(" ")
            bbox = [float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4])]
            bboxes.append(bbox)
        return bboxes


def _bbox_to_img_area_ratio_lt_thresh(
    bbox_width, bbox_height, img_width, img_height, thresh
):
    assert thresh in range(0, 1), "Threshhold must be a value between [0,1]"

    if img_width < 0 or img_height < 0:
        img_size_neg_error_msg = (
            "Image width and height must be positive values. "
            + f"Actual values: width={img_width} | height={img_height}"
        )
        raise ValueError(img_size_neg_error_msg)

    if bbox_width < 0 or bbox_height < 0:
        bbox_neg_error_msg = (
            "Bbox width and height must be positive values. "
            + f"Actual values: width={bbox_width} | height={bbox_height}"
        )
        raise ValueError(bbox_neg_error_msg)
    bbox_area = bbox_width * bbox_height
    img_area = img_width * img_height

    assert not bbox_area > img_area, (
        "Bounding box area is greater than image. "
        + f"Actual values: bbox area={bbox_area} | img area={img_area}"
    )

    return (bbox_area / img_area) < thresh


def main(path, labels_filter, force_filtering=False):
    # TODO: #14 read name, sample and background from config file
    name = ["train2017", "val2017"]
    sample = [1, 1]
    numBackground = [1500, 100]
    if isinstance(name, list):
        for n, s, b in zip(name, sample, numBackground):
            appendix = str(s) + "_6cl"
            _filter_labels(
                labels_filter=labels_filter,
                path=path,
                name=n,
                appendix=appendix,
                num_background=b,
                sample=s,
                reset_label_ids=True,
                force_filtering=force_filtering,
            )
    else:
        appendix = str(sample)
        _filter_labels(
            labels_filter,
            path,
            name,
            appendix,
            numBackground,
            sample,
            force_filtering=force_filtering,
            reset_label_ids=True,
        )


if __name__ == "__main__":
    path = "./OTLabels/data/coco"
    labelsFilter = "./OTLabels/labels_CVAT.txt"
    main(path, labelsFilter)
