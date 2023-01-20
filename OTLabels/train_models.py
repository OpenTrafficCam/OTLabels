import argparse
from pathlib import Path
from typing import Union

import yolo
from config import CONFIG
from helpers.files import get_files, is_in_format
from helpers.log import log
from ultralytics import YOLO


def main(
    paths: Union[list, str, Path],
    filetypes: list = CONFIG["FILETYPES"]["IMG"],
    model=None,
    weights: str = CONFIG["DETECT"]["YOLO"]["WEIGHTS"],
    conf: float = CONFIG["DETECT"]["YOLO"]["CONF"],
    iou: float = CONFIG["DETECT"]["YOLO"]["IOU"],
    size: int = CONFIG["DETECT"]["YOLO"]["IMGSIZE"],
    chunksize: int = CONFIG["DETECT"]["YOLO"]["CHUNKSIZE"],
    normalized: bool = CONFIG["DETECT"]["YOLO"]["NORMALIZED"],
    overwrite: bool = CONFIG["DETECT"]["OVERWRITE"],
    ot_labels_enabled: bool = False,
    debug: bool = CONFIG["DETECT"]["DEBUG"],
):
    log.info("Start detection")
    if debug:
        log.setLevel("DEBUG")
        log.debug("Debug mode on")

    if not model:
        yolo_model = yolo.loadmodel(weights, conf, iou)
    else:
        yolo_model = model
        yolo_model.conf = conf
        yolo_model.iou = iou
    log.info("Model prepared")

    files = get_files(paths=paths, filetypes=filetypes)
    img_files = _split_to_video_img_paths(files)
    log.info("Files splitted in videos and images")

    log.info(f"Try detecting {len(img_files)} images")
    img_file_chunks = _create_chunks(img_files, chunksize)
    detections_img_file_chunks = yolo.detect_images(
        file_chunks=img_file_chunks,
        model=yolo_model,
        weights=weights,
        conf=conf,
        iou=iou,
        size=size,
        chunksize=chunksize,
        normalized=normalized,
        ot_labels_enabled=ot_labels_enabled,
    )
    log.info("Images detected")

    if ot_labels_enabled:
        return detections_img_file_chunks
    # for img_file, detection in zip(img_files, detections_img_file_chunks):
    #     write(detection, img_file)


def _split_to_video_img_paths(
    files,
    video_formats=CONFIG["FILETYPES"]["VID"],
    img_formats=CONFIG["FILETYPES"]["IMG"],
):
    """Divide a list of files in video files and other files.

    Args:
        file_paths (list): The list of files.
        vidoe_formats

    Returns:
        [list(str), list{str)] : list of video paths and list of images paths
    """
    video_files, img_files = [], []
    for file in files:
        if is_in_format(file, video_formats):
            video_files.append(file)
        elif is_in_format(file, img_formats):
            img_files.append(file)
        else:
            raise FormatNotSupportedError(
                f"The format of path is not supported ({file})"
            )
    return video_files, img_files


class FormatNotSupportedError(Exception):
    pass


def _create_chunks(files, chunksize):
    if chunksize == 0:
        return files
    chunk_starts = range(0, len(files), chunksize)
    return [files[i : i + chunksize] for i in chunk_starts]


def parse():
    parser = argparse.ArgumentParser(description="Detect objects in videos or images")
    parser.add_argument(
        "-p",
        "--paths",
        nargs="+",
        type=str,
        help="Path/list of paths to image or video or folder containing videos/images",
        required=True,
    )
    parser.add_argument(
        "-f",
        "--filetypes",
        type=str,
        nargs="+",
        help="Filetypes of files in folders to select for detection",
        required=False,
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        help="Name of weights from PyTorch hub or Path to weights file",
        required=False,
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Logging in debug mode"
    )
    return parser.parse_args()


def train_basic_model():
    model = YOLO("yolov8n.pt")  # pass any model type
    model.train(data="train-config.yaml", epochs=5)


if __name__ == "__main__":
    # kwargs = vars(parse())
    # log.info("Starting detection from command line")
    # log.info(f"Arguments: {kwargs}")
    # main(**kwargs)
    # log.info("Finished detection from command line")
    train_basic_model()
