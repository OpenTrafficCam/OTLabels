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

# TODO: docstrings in get_coco_annotation_files

from pathlib import Path
import shutil
import urllib.request


def _downloadAnnotations(URLFile, cocoPath):
    with open(URLFile, "r") as f:
        URLs = f.readlines()

    if not Path(cocoPath).exists():
        Path(cocoPath).mkdir()

    for URL in URLs:
        print('Download and unzip annotation files to "' + cocoPath + '"')

        annoFile = cocoPath + "/annotations.zip"
        urllib.request.urlretrieve(URL, annoFile)

        _unzip(annoFile, cocoPath)


def _unzip(file, dir):
    file = Path(file)
    shutil.unpack_archive(file, dir)
    Path.unlink(file)


if __name__ == "__main__":
    path = "OTLabels/data/coco"
    URLFile = "OTLabels/coco_annotation_json_URLs.txt"

    _downloadAnnotations(URLFile, path)
