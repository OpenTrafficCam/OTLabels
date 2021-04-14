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
        print("Download and unzip image files to \"" + imagePath + "\"...", end="")
        dataVersion = URL.split("/")[-1]
        imageFile = imagePath + "/" + dataVersion
        urllib.request.urlretrieve(URL, imageFile)

        _unzip(imageFile, imagePath)

        print("Done!")


def _downloadAnnotations(annURL, annPath):
    print("Download and unzip anotation files to \"" + annPath + "/coco\"...", end="")
    dataVersion = annURL.split("/")[-1]
    annFile = annPath + "/" + dataVersion
    urllib.request.urlretrieve(annURL, annFile)

    _unzip(annFile, annPath)

    print("Done!")


def _downloadCocoData(imageURLs, annURL, path):
    imagePath = path + "/coco/images"

    _downloadAnnotations(annURL, path)
    _downloadImages(imageURLs, imagePath)


if __name__ == "__main__":
    imageURLs = "OTLabels/coco_image_URLs.txt"
    annURL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip"
    path = "OTLabels/data"

    _downloadCocoData(imageURLs, annURL, path)
