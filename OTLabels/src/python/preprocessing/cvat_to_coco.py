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

# TODO: docstrings in cvat_to_coco

from pathlib import Path
import shutil
from pycocotools.coco import COCO
import pandas as pd


def _getCocoCats(catFile, annFile):
    cats = pd.read_csv(catFile)['Cat'].tolist()
    coco = COCO(annFile)
    catIds = coco.getCatIds(catNms=cats)
    labelsCoco = pd.DataFrame(list(zip(cats, catIds)), columns=["Cat", "CatId"])
    return labelsCoco


def _fileList(file, suffix):
    file = Path(file)
    dir = file.with_suffix("")
    files = dir.glob("obj_train_data/*.{}".format(suffix))
    return [str(file) for file in files]


def _unzip(file):
    file = Path(file)
    dir = file.with_suffix("")
    shutil.unpack_archive(file, dir)
    imageFiles = _fileList(file, "png")
    annFiles = _fileList(file, "txt")
    return imageFiles, annFiles, dir


def _copyFiles(sourceFiles, destPath):
    if not Path(destPath).exists():
        Path(destPath).mkdir()
    for f in sourceFiles:
        shutil.copy(f, destPath)


def _createLabelDict(labelsCVAT, labelsCOCO):
    labels = pd.merge(labelsCVAT, labelsCOCO,
                      how="inner",
                      on="Cat",
                      suffixes=["_CVAT", "_COCO"])
    labelDict = {}
    for i in labels["Cat"]:
        labelDict.update({labels.loc[labels["Cat"] == i, "CatId_CVAT"].values[0]:
                          labels.loc[labels["Cat"] == i, "CatId_COCO"].values[0]})
    return labelDict


def _copyFilesConvert(annFiles, annPath, labelsCVAT, labelsCOCO):
    if not Path(annPath).exists():
        Path(annPath).mkdir()

    labelDict = _createLabelDict(labelsCVAT, labelsCOCO)

    for f in annFiles:
        fileName = f.split("\\")[-1]
        fileLabels = pd.read_csv(f, header=None, sep=" ")
        fileLabels[0] = fileLabels[0].map(labelDict)
        fileLabels = fileLabels.dropna()
        fileLabels[0] = fileLabels[0].astype(int)
        fileLabels.to_csv(annPath + "/" + fileName,
                          header=False,
                          sep=" ",
                          index=False,
                          line_terminator="\n")


def _fileStructure(cvatFile, destPath, labelsCVAT, labelsCOCO, name):
    imageFiles, annFiles, sourcepath = _unzip(cvatFile)
    imagePath = destPath + "/images/" + name
    annPath = destPath + "/labels/" + name

    _copyFiles(imageFiles, imagePath)

    _copyFilesConvert(annFiles, annPath, labelsCVAT, labelsCOCO)
    shutil.rmtree(sourcepath)
    return annFiles


def _cvatToCoco(destPath, cvatFile, labelsCVAT, labelsCOCO, name):
    assert len(labelsCVAT) == len(labelsCOCO), "CVAT Labels and COCO Labels differ!"

    _fileStructure(cvatFile, destPath, labelsCVAT, labelsCOCO, name)


if __name__ == "__main__":
    destPath = "D:/OTC/OTLabels/OTLabels/data/coco"
    annFile = "D:/OTC/OTLabels/OTLabels/data/coco/annotations/instances_val2017.json"
    catFile = "D:/OTC/OTLabels/OTLabels/labels_CVAT.txt"
    cvatFile = "C:/Users/MichaelHeilig/Downloads/task_radeberg_fr20_2020-" \
        "02-20_12-15-00-2021_04_07_16_12_13-yolo 1.1.zip"
    name = "radeberger-00"

    labelsCVAT = pd.read_csv(catFile)
    labelsCOCO = _getCocoCats(catFile, annFile)

    _cvatToCoco(destPath, cvatFile, labelsCVAT, labelsCOCO, name)
