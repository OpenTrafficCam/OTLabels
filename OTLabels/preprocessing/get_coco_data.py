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


def _downloadImages(imageURLs, imagePath):
    with open(imageURLs, "r") as f:
        URLs = f.read().splitlines()

    if not Path(imagePath).exists():
        Path(imagePath).mkdir()

    for URL in URLs:
        if URL[0] == "#":
            continue
        elif URL[0] != "h":
            print("The provided URL is not valid!")
            continue
        print('Download and unzip image files to "' + imagePath + '"...')
        dataVersion = URL.split("/")[-1]
        imageFile = imagePath + "/" + dataVersion
        urllib.request.urlretrieve(URL, imageFile)

        _unzip(imageFile, imagePath)

        print("Done!")


def _downloadAnnotations(annURLs, annPath):
    with open(annURLs, "r") as f:
        URLs = f.read().splitlines()

    for URL in URLs:
        if URL[0] == "#":
            continue
        elif URL[0] != "h":
            print("The provided URL is not valid!")
            continue
    print('Download and unzip annotation files to "' + annPath + '/coco"...')
    dataVersion = URL.split("/")[-1]
    annFile = annPath + "/" + dataVersion
    urllib.request.urlretrieve(URL, annFile)

    _unzip(annFile, annPath)

    print("Done!")


def _downloadCocoData(imageURLs, annURL, path):
    imagePath = path + "/coco/images"

    _downloadAnnotations(annURL, path)
    _downloadImages(imageURLs, imagePath)


if __name__ == "__main__":
    imageURLs = "OTLabels/coco_image_URLs.txt"
    annURL = "OTLabels/coco_annotation_URLs.txt"
    path = "OTLabels/data"

    _downloadCocoData(imageURLs, annURL, path)
