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
import os
import shutil
import random


def _resetLabels(labels):
    labelDict = {}
    row = 0
    for i in labels["Cat"]:
        labelDict.update({labels.loc[labels["Cat"] == i, "CatId"].values[0]: row})
        row = row + 1
    return labelDict


def _fileList(file, suffix):
    file = Path(file)
    dir = file.with_suffix("")
    if suffix == "":
        files = dir.glob("*")
    else:
        files = dir.glob("*.{}".format(suffix))
    return [str(file) for file in files]


def _filterLabels(labelsFilter, path, name, numBackground, newTrain=False, sample=1.0):
    sourcePath = path + "/labels/" + name
    imagePath = path + "/images/" + name
    destPathLabels = path + "/labels/" + name + "_filtered"
    destPathImgs = path + "/images/" + name + "_filtered"

    img_type = _fileList(imagePath, "")[0].split("\\")[-1].split(".")[-1].lower()

    if Path(destPathLabels).exists():
        shutil.rmtree(destPathLabels)
    if Path(destPathImgs).exists():
        shutil.rmtree(destPathImgs)

    Path(destPathLabels).mkdir()
    Path(destPathImgs).mkdir()

    labels = pd.read_csv(labelsFilter)
    annFiles = _fileList(sourcePath, "txt")

    if newTrain:
        labelDict = _resetLabels(labels)

    print("Filter files in \"" + sourcePath + "\" by labels " +
          ', '.join(str(e) for e in labels["Cat"].tolist()) + "...", end="")

    imageList = []
    imageListSource = []
    n = 0
    for f in annFiles:

        write = random.uniform(0, 1) < sample

        fileName = f.split("\\")[-1]
        imageName = fileName.split(".")[0] + "." + img_type
        if os.stat(f).st_size > 0:
            fileLabels = pd.read_csv(f, header=None, sep=" ")
        else:
            continue

        fileLabels = fileLabels[fileLabels[0].isin(labels["CatId"])]

        if len(fileLabels) > 0:
            if newTrain:
                fileLabels[0] = fileLabels[0].map(labelDict)
                fileLabels = fileLabels.dropna()
                fileLabels[0] = fileLabels[0].astype(int)
            if write:
                fileLabels.to_csv(destPathLabels + "/" + fileName,
                                  header=False,
                                  sep=" ",
                                  index=False,
                                  line_terminator="\n")
                imageList.append(destPathImgs + "/" + imageName)
                imageListSource.append(imagePath + "/" + imageName)
        else:
            if n < numBackground:
                open(destPathLabels + "/" + fileName, 'a').close()
                imageList.append(destPathImgs + "/" + imageName)
                imageListSource.append(imagePath + "/" + imageName)
            n = n + 1
            continue

    with open(path + "/" + name + "_filtered.txt", "w") as f:
        f.write('\n'.join(imageList))

    for img in imageListSource:
        shutil.copy(img, destPathImgs)

    print("Done!")


if __name__ == "__main__":
    path = "D:/OTC/OTLabels/OTLabels/data/coco"
    name = ["train2017", "val2017"]
    sample = [0.08, 1]
    labelsFilter = "D:/OTC/OTLabels/OTLabels/labels_CVAT.txt"
    numBackground = 500
    if isinstance(name, list):
        for n, s in zip(name, sample):
            _filterLabels(labelsFilter, path, n, numBackground, s, newTrain=True)
    else:
        _filterLabels(labelsFilter, path, name, numBackground, sample, newTrain=True)
