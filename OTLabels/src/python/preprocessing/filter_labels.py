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
from pycocotools.coco import COCO
import pandas as pd
import os


def _getCocoCats(catFile, annFile):
    with open(catFile, "r") as cat:
        cats = cat.read().splitlines()
    coco = COCO(annFile)
    catIds = coco.getCatIds(catNms=cats)
    labelsCoco = pd.DataFrame(list(zip(cats, catIds)), columns=["Cat", "CatId"])
    return labelsCoco


def _fileList(file, suffix):
    file = Path(file)
    dir = file.with_suffix("")
    if suffix == "":
        files = dir.glob("*")
    else:
        files = dir.glob("*.{}".format(suffix))
    return [str(file) for file in files]


def _filterLabels(labelsFilter, path, name, annFile):
    sourcePath = path + "/labels/" + name
    imagePath = path + "/images/" + name
    destPath = path + "/labels/" + name + "_filtered"

    img_type = _fileList(imagePath, "")[0].split("\\")[-1].split(".")[-1].lower()

    if not Path(destPath).exists():
        Path(destPath).mkdir()

    labels = _getCocoCats(labelsFilter, annFile)
    annFiles = _fileList(sourcePath, "txt")

    print("Filter files in \"" + sourcePath + "\" by labels " +
          ', '.join(str(e) for e in labels["Cat"].tolist()) + "...", end="")

    imageList = []
    for f in annFiles:
        fileName = f.split("\\")[-1]
        imageName = fileName.split(".")[0] + "." + img_type
        if os.stat(f).st_size > 0:
            fileLabels = pd.read_csv(f, header=None, sep=" ")
        else:
            continue
        fileLabels = fileLabels[fileLabels[0].isin(labels["CatId"])]

        if len(fileLabels) > 0:
            fileLabels.to_csv(destPath + "/" + fileName,
                              header=False,
                              sep=" ",
                              index=False,
                              line_terminator="\n")
            imageList.append(destPath + "/" + imageName)
        else:
            continue

    with open(path + "/" + name + "_filtered.txt", "w") as f:
        f.write('\n'.join(imageList))
    print("Done!")


if __name__ == "__main__":
    path = "D:/OTC/OTLabels/OTLabels/data/coco"
    name = ["train2017", "val2017"]
    annFile = "D:/OTC/OTLabels/OTLabels/data/coco/annotations/instances_val2017.json"
    labelsFilter = "D:/OTC/OTLabels/OTLabels/label_filter.txt"
    if isinstance(name, list):
        for n in name:
            _filterLabels(labelsFilter, path, n, annFile)
    else:
        _filterLabels(labelsFilter, path, name, annFile)
