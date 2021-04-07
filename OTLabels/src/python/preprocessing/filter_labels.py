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
from shutil import copytree


def _copyFiles(sourcePath, destPath):
    copytree(sourcePath, destPath)


def _files(path):
    dir = path.with_suffix("")
    files = dir.glob("org/obj_train_data/*.txt")

    return files


def _readFile(file):
    with open(file, "r") as f:
        return f.readlines()
        

def _filterRows(file, lines, labels=[]):
    if not isinstance(labels, list):
        labels = [labels]

    with open(file, "w") as f:
        for line in lines:
            label = line.split()[0]
            if label in labels:
                f.write(line)


def _filterLabels(labels, path):
    sourcePath = Path(path)
    destPath = Path(path + "_org")

    _copyFiles(sourcePath, destPath)

    sourceFiles = _files(sourcePath)
    



if __name__ == "__main__":
    labels = [0, 1, 2, 5, 7]
    path = "D:/OTC/OTLabels/OTLabels/data/retraining/Radeberg"

    print("Filter files in \"" + path + "\" by labels " + ', '.join(str(e) for e in labels))

    