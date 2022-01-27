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
from tqdm import tqdm


def _resetLabels(labels):
    labelDict = {}
    row = 0
    for i in labels["Cat"]:
        labelDict.update({labels.loc[labels["Cat"] == i, "CatId"].values[0]: row})
        row = row + 1
    return labelDict


def _fileList(file, suffix):
    file = Path(file)
    print('Reading "{file}\\*{suffix}" files...'.format(file=file, suffix=suffix))
    dir = file.with_suffix("")
    if suffix == "":
        files = dir.glob("*")
    else:
        files = dir.glob("*.{}".format(suffix))
    return [str(file) for file in files]


def _filter_labels(
    labelsFilter,
    path,
    name,
    appendix,
    numBackground,
    sample=1.0,
    resetLabelIds=False,
):
    sourcePath = Path(path, "labels/" + name)
    imagePath = Path(path, "images/" + name)
    destPathLabels = Path(path, "labels/" + name + "_filtered_" + appendix)
    destPathImgs = Path(path, "images/" + name + "_filtered_" + appendix)
    destPathImgsRelative = "./images/" + name + "_filtered_" + appendix

    # TODO: Path(_file_list(image_path, "")[0].suffix)
    imgType = Path(_fileList(imagePath, "")[0]).suffix.lower()

    # Remove existing filtered data
    if Path(destPathLabels).exists():
        shutil.rmtree(destPathLabels)
    if Path(destPathImgs).exists():
        shutil.rmtree(destPathImgs)

    Path(destPathLabels).mkdir(parents=True)
    Path(destPathImgs).mkdir(parents=True)

    labels = pd.read_csv(labelsFilter)
    annFiles = _fileList(sourcePath, "txt")

    if resetLabelIds:
        labelDict = _resetLabels(labels)

    print(
        'Filter files in "'
        + str(sourcePath)
        + '" by labels '
        + ", ".join(str(e) for e in labels["Cat"].tolist())
        + "...",
    )

    imageList = []
    imageListSource = []
    n = 0
    for f in tqdm(annFiles):

        write = random.uniform(0, 1) < sample

        fileName = Path(f).name
        imageName = Path(fileName).stem + imgType
        if os.stat(f).st_size > 0:
            fileLabels = pd.read_csv(f, header=None, sep=" ")
        else:
            continue

        fileLabels = fileLabels[fileLabels[0].isin(labels["CatId"])]

        if len(fileLabels) > 0:
            if resetLabelIds:
                fileLabels[0] = fileLabels[0].map(labelDict)
                fileLabels = fileLabels.dropna()
                fileLabels[0] = fileLabels[0].astype(int)
            if write:
                fileLabels.to_csv(
                    Path(destPathLabels, fileName),
                    header=False,
                    sep=" ",
                    index=False,
                    line_terminator="\n",
                )
                imageList.append(str(Path(destPathImgsRelative, imageName)))
                imageListSource.append(str(Path(imagePath, imageName)))
        else:
            if n < numBackground:
                open(Path(destPathLabels, fileName), "a").close()  # NOTE: Reason?
                imageList.append(str(Path(destPathImgsRelative, imageName)))
                imageListSource.append(str(Path(imagePath, imageName)))
            n = n + 1
            continue

    file_filteredlabels = Path(path, name + "_filtered_" + appendix + ".txt")
    print(
        "Writing file with filtered labels to {path} ...".format(
            path=file_filteredlabels
        )
    )

    with open(file_filteredlabels, "w") as f:
        f.write("\n".join(imageList))

    print(
        "Copying {num_imgs} images to {path} ...".format(
            num_imgs=len(imageList), path=destPathImgs
        )
    )
    for img in tqdm(imageListSource):
        shutil.copy(img, destPathImgs)

    print("Done!")


if __name__ == "__main__":
    path = "./OTLabels/data/coco"
    labelsFilter = "./OTLabels/labels_CVAT.txt"
    # TODO: #14 read name, sample and background from config file
    name = ["train2017", "val2017"]
    sample = [1, 1]
    numBackground = [1500, 100]
    if isinstance(name, list):
        for n, s, b in zip(name, sample, numBackground):
            appendix = str(s) + "_6cl"
            _filterLabels(labelsFilter, path, n, appendix, b, s, resetLabelIds=True)
    else:
        appendix = str(sample)
        _filterLabels(
            labelsFilter,
            path,
            name,
            appendix,
            numBackground,
            sample,
            resetLabelIds=True,
        )
