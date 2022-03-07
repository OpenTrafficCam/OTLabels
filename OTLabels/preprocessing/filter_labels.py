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
from typing import Union
from typing import Tuple
from typing import List

from pathlib import Path
from PIL import Image
import shutil
import random
from tqdm import tqdm

from ..helpers.files import get_yolov5_img_path_from_ann_path
from ..helpers.files import write_yolov5_anns_to_file
from ..helpers.files import read_cvat_labels_file


def _reset_labels(labels: dict):
    label_dict = {}

    for idx, class_id in enumerate(labels.values()):
        label_dict.update({class_id: idx})

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
    labels_filter: Union[str, Path],
    path: Union[str, Path],
    name: str,
    appendix: str,
    num_background: int,
    force_filtering: bool,
    normalized,
    sample=1.0,
    lower_thresh: float = 0,
    upper_thresh: float = 1,
    apply_thresh_filter: bool = False,
    reset_label_ids: bool = False,
    discard_img_above_thresh: bool = False,
    keep_discarded_imgs: bool = False,
):
    labels_dir = Path(path, f"labels/{name}")
    img_dir = Path(path, f"images/{name}")
    dest_dir_labels = Path(path, f"labels/{name}_filtered_{appendix}")
    dest_dir_imgs = Path(path, f"images/{name}_filtered_{appendix}")
    dest_dir_imgs_relative = f"./images/{name}_filtered_{appendix}"
    dest_dir_discarded_imgs = Path(path, f"images/{name}_filtered_{appendix}_discarded")

    img_type = Path(_file_list(img_dir, "")[0]).suffix.lower()

    if not force_filtering and dest_dir_labels.exists() and dest_dir_imgs.exists():
        print(
            "Filtered data already exists. To force filtering set force_filtering FLAG "
            + f'to True or delete these folders: "{dest_dir_labels}", "{dest_dir_imgs}"'
        )
        return

    # Remove existing filtered data
    if dest_dir_labels.exists():
        shutil.rmtree(dest_dir_labels)
    if dest_dir_imgs.exists():
        shutil.rmtree(dest_dir_imgs)
    if dest_dir_discarded_imgs.exists():
        shutil.rmtree(dest_dir_discarded_imgs)

    Path(dest_dir_labels).mkdir(parents=True)
    Path(dest_dir_imgs).mkdir(parents=True)
    if keep_discarded_imgs:
        dest_dir_discarded_imgs.mkdir(parents=True)

    class_labels = read_cvat_labels_file(labels_filter, delimiter=",")
    ann_files = _file_list(labels_dir, "txt")

    if reset_label_ids:
        label_dict = _reset_labels(class_labels)

    print(
        f"Filter files in {labels_dir} by labels "
        + ", ".join(str(class_name) for class_name in class_labels)
        + "..."
    )

    rel_img_paths = []
    src_img_paths = []
    discarded_imgs = []
    n = 0
    for ann_file in tqdm(ann_files):

        write = random.uniform(0, 1) < sample

        file_name = Path(ann_file).name
        img_name = Path(file_name).stem + img_type

        if _has_data(ann_file):
            bbox_anns = _get_bboxes(ann_file)

        bbox_anns_filtered = [
            bbox for bbox in bbox_anns if _is_in_cls_labels(class_labels, bbox)
        ]

        if len(bbox_anns_filtered) > 0:
            dont_write = False

            if reset_label_ids:
                bbox_anns_filtered = _change_bboxes_cls_ids(
                    label_dict, bbox_anns_filtered
                )
            if write:
                if apply_thresh_filter:
                    if normalized:
                        img_width = 1
                        img_height = 1
                    else:
                        img_path = get_yolov5_img_path_from_ann_path(ann_file, img_type)
                        img_width, img_height = Image.open(img_path).size

                    (
                        thresh_filtered_labels,
                        discarded,
                    ) = _filter_bboxes_with_bbox_img_ratio(
                        anns=bbox_anns_filtered,
                        img_width=img_width,
                        img_height=img_height,
                        lower_thresh=lower_thresh,
                        upper_thresh=upper_thresh,
                    )
                    assert len(thresh_filtered_labels) + len(discarded) == len(
                        bbox_anns_filtered
                    )

                    if discard_img_above_thresh:
                        # set flag to True if img contains bboxes
                        # greater than upper_thresh
                        dont_write = True in [
                            bbox_img_ratio > upper_thresh
                            for bbox_img_ratio in discarded
                        ]

                    if not dont_write:
                        if len(thresh_filtered_labels) == 0:
                            # use as background image if no bboxes left after filtering
                            n = n + 1

                        write_yolov5_anns_to_file(
                            anns=thresh_filtered_labels,
                            dest=Path(dest_dir_labels, file_name),
                        )
                else:
                    write_yolov5_anns_to_file(
                        anns=bbox_anns_filtered, dest=Path(dest_dir_labels, file_name)
                    )

                if dont_write:
                    if keep_discarded_imgs:
                        discarded_imgs.append(Path(img_dir, img_name))
                else:
                    rel_img_paths.append(
                        "./" + str(Path(dest_dir_imgs_relative, img_name))
                    )
                    src_img_paths.append(Path(img_dir, img_name))
        else:
            if n < num_background:
                open(Path(dest_dir_labels, file_name), "a").close()  # Create empty ann
                rel_img_paths.append(f"./{Path(dest_dir_imgs_relative, img_name)}")
                src_img_paths.append(Path(img_dir, img_name))
            n = n + 1

    file_filtered_labels = Path(path, f"{name}_filtered_{appendix}.txt")
    print(f"Writing file with filtered labels to {file_filtered_labels} ...")

    with open(file_filtered_labels, "w") as ann_file:
        ann_file.write("\n".join(rel_img_paths))

    print(f"Copying {len(rel_img_paths)} images to {dest_dir_imgs} ...")
    for src_img in tqdm(src_img_paths):
        shutil.copy(src_img, dest_dir_imgs)

    if keep_discarded_imgs:
        print(
            f"Copying {len(discarded_imgs)} discarded images to {dest_dir_discarded_imgs}"
            + "..."
        )
        for src_disc_img in tqdm(discarded_imgs):
            shutil.copy(src_disc_img, dest_dir_discarded_imgs)

    print("Done!")


def _is_in_cls_labels(class_labels: dict, bbox: list):
    CLASS_ID = 0
    for cls_id in class_labels.values():
        if bbox[CLASS_ID] == cls_id:
            return True

    return False


def _change_bboxes_cls_ids(cls_mapper: dict, bboxes):
    """
    Change the bounding boxes class ids.

    Assumes that the bboxes' class ids are specified in `cls_mapper`,ß
    otherwise raise `KeyError`.
    """
    filtered = []
    for bbox in bboxes:
        filtered.append(_change_cls_id(bbox, cls_mapper=cls_mapper))
    return filtered


def _change_cls_id(bbox: list, cls_mapper: dict):
    """
    Change a bbox's class id if it exists in the cls_mapper, otherwise raise `KeyError`.
    """
    cls_id, x, y, w, h = bbox
    return [cls_mapper[cls_id], x, y, w, h]


def _filter_bboxes_with_bbox_img_ratio(
    anns: list,
    img_width: int,
    img_height: int,
    lower_thresh: float = 0,
    upper_thresh: float = 1,
) -> Tuple[list, List[float]]:
    """
    Filters out bounding boxes that are not within the specified threshold.

    Args:
        anns (list): list of bounding boxes in [class, x, y, w, h] format.
        img_width (int): set to 1, if working with normalized bboxes.
        img_height (int): set to 1, if working with normalized bboxes.
        lower_thresh (float): bboxes with a bbox-img ratio below this thresh will be
        discarded.
        upper_thresh (float): bboxes with a bbox_img ratio above this thresh will be
        discarded.

    Returns:
        A tuple (keep, discarded) where `keep` denotes a list of bboxes within the
        threshold, and `discarded` denotes a list bbox-img-ratio of discarded bboxes.
    """
    # get bboxes
    if len(anns) == 0:
        return []

    filtered_bbox_anns = []
    bbox_img_ratios_of_discarded = []
    for bbox in anns:
        _cls, x, y, w, h = bbox
        bbox_to_img_area_ratio = _calc_bbox_to_img_ratio(
            bbox_width=w, bbox_height=h, img_width=img_width, img_height=img_height
        )
        if _is_bbox_to_img_ratio_in_thresh(
            bbox_to_img_ratio=bbox_to_img_area_ratio,
            lower_thresh=lower_thresh,
            upper_thresh=upper_thresh,
        ):
            filtered_bbox_anns.append([int(_cls), x, y, w, h])
        else:
            bbox_img_ratios_of_discarded.append(bbox_to_img_area_ratio)
    return filtered_bbox_anns, bbox_img_ratios_of_discarded


def _get_cvat_yolo_ann_path_from_img_path(img_path: Path):
    return Path(img_path.parent.with_stem("labels"), f"{img_path.stem}.txt")


def _get_bboxes(label_path):
    """
    Reads and parses the bounding boxes from a YOLOv5 annotation text file.

    The text file should contain bboxes in the form of `class x y w h\\n` where each
    value is separated by a whitespace and a newline marks a new bounding box.

    Args:
        label_path(str|PosixPath): Path to text file in YOLOv5 annotation format.

    Returns:
        A list containing the bounding box values in the form of [cls, x, y, w, h].
    """
    with open(label_path, "r", errors="ignore") as f:
        lines = [line.rstrip("\n") for line in f.readlines()]
        bboxes = []
        for line in lines:
            vals = line.split(" ")
            bbox = [
                int(vals[0]),
                float(vals[1]),
                float(vals[2]),
                float(vals[3]),
                float(vals[4]),
            ]
            bboxes.append(bbox)
        return bboxes


def _calc_bbox_to_img_ratio(
    bbox_width: float, bbox_height: float, img_width: float, img_height: float
):
    if img_width <= 0 or img_height <= 0:
        img_size_neg_error_msg = (
            "Image width and height must be positive values greater than 0. "
            + f"Actual values: width={img_width} | height={img_height}"
        )
        raise ValueError(img_size_neg_error_msg)

    if bbox_width < 0 or bbox_height < 0:
        bbox_neg_error_msg = (
            "Bbox width and height must be positive values greater than 0. "
            + f"Actual values: width={bbox_width} | height={bbox_height}"
        )
        raise ValueError(bbox_neg_error_msg)

    bbox_area = bbox_width * bbox_height
    img_area = img_width * img_height

    if bbox_area > img_area:
        err_msg = (
            "Bounding box area is greater than image. "
            + f"Actual values: bbox area={bbox_area} | img area={img_area}"
        )
        raise ValueError(err_msg)

    bbox_to_img_ratio = bbox_area / img_area
    assert bbox_to_img_ratio <= 1 and bbox_to_img_ratio >= 0
    return bbox_to_img_ratio


def _is_bbox_to_img_ratio_in_thresh(
    bbox_to_img_ratio: float,
    lower_thresh: float,
    upper_thresh: float,
):
    assert (
        lower_thresh <= upper_thresh
        and lower_thresh >= 0
        and lower_thresh <= 1
        and upper_thresh >= 0
        and upper_thresh <= 1
    ), (
        "Condition that 0 <= 'lower_thresh <= upper_thresh' <= 1 not fulfilled.\n"
        + f"Actual value: lower_thresh = {lower_thresh}, upper_thresh = {upper_thresh}"
    )

    return lower_thresh <= bbox_to_img_ratio and bbox_to_img_ratio <= upper_thresh


def main(
    path: Union[str, Path],
    labels_filter: Union[str, Path],
    normalized: bool,
    lower_thresh: float,
    upper_thresh: float,
    apply_thresh_filter: bool,
    discard_img_above_thresh: bool,
    keep_discarded_imgs: bool = False,
    force_filtering: bool = False,
):
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
                force_filtering=force_filtering,
                normalized=normalized,
                lower_thresh=lower_thresh,
                upper_thresh=upper_thresh,
                apply_thresh_filter=apply_thresh_filter,
                reset_label_ids=True,
                discard_img_above_thresh=discard_img_above_thresh,
                keep_discarded_imgs=keep_discarded_imgs,
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
