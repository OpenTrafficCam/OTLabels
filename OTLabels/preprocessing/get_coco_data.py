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

# TODO: docstrings in get_coco_data

from pathlib import Path
import shutil
import urllib.request


def _unzip(file, dir=""):
    file = Path(file)
    if dir == "":
        dir = file.with_suffix("")
    shutil.unpack_archive(file, dir)
    Path.unlink(file)


def _download_images(imageURLs, imagePath):
    with open(imageURLs, "r") as f:
        URLs = f.read().splitlines()

    if not Path(imagePath).exists():
        Path(imagePath).mkdir(parents=True)

    for URL in URLs:
        if URL[0] == "#":
            continue
        elif URL[0] != "h":
            print("The provided URL is not valid!")
            continue
        print('Download and unzip image files to "' + str(imagePath) + '"...')
        dataVersion = URL.split("/")[-1]
        imageFile = Path(imagePath, dataVersion)
        urllib.request.urlretrieve(URL, imageFile)

        _unzip(imageFile, imagePath)

        print("Done!")


def _download_annotations(ann_urls, ann_path):
    if not Path(ann_path).exists() and Path(ann_path).is_dir():
        Path(ann_path).mkdir(parents=True)

    with open(ann_urls, "r") as f:
        URLs = f.read().splitlines()

    for URL in URLs:
        if URL[0] == "#":  # NOTE: Why?? What is checked here
            continue
        elif URL[0] != "h":
            # TODO: check if it's starts with h as in http but better to use starts with
            # or some regex to check if the whole URL exists
            print("The provided URL is not valid!")
            continue
        # NOTE: No break condition if not valid URL

    print('Download and unzip annotation files to "' + str(ann_path) + "...")
    dataVersion = URL.split("/")[-1]
    annFile = Path(ann_path, dataVersion)
    urllib.request.urlretrieve(URL, annFile)

    _unzip(annFile, ann_path)

    print("Done!")


def main(image_urls, ann_url, save_path, force_download=False):
    image_path = Path(save_path, "coco/images")

    if force_download:
        _download_annotations(ann_url, save_path)
        _download_images(image_urls, image_path)
    else:
        if not Path(save_path, "coco/annotations").exists():
            _download_annotations(ann_url, save_path)

        elif not image_path.exists():
            _download_images(image_urls, image_path)


if __name__ == "__main__":
    image_urls = "OTLabels/coco_image_URLs.txt"
    ann_url = "OTLabels/coco_annotation_URLs.txt"
    path = "OTLabels/data"
    main(image_urls, ann_url, path)
