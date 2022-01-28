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


def _unzip(file, dir):
    file = Path(file)
    shutil.unpack_archive(file, dir)
    Path.unlink(file)


def main(url_file, coco_path):
    with open(url_file, "r") as f:
        urls = f.readlines()

    if not Path(coco_path).exists():
        Path(coco_path).mkdir()

    for url in urls:
        print(f'Download and unzip annotation files to "{coco_path}"')

        ann_file = Path(coco_path, "annotations.zip")
        urllib.request.urlretrieve(url, ann_file)

        _unzip(ann_file, coco_path)


if __name__ == "__main__":
    coco_path = "OTLabels/data/coco"
    url_file = "OTLabels/coco_annotation_json_URLs.txt"
    main(url_file, coco_path)
