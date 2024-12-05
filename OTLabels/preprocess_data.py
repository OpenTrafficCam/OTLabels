"""Preprocess image data for annotation in CVAT"""

import itertools
import random
from datetime import datetime
from pathlib import Path
from typing import Iterator

import fiftyone

from OTLabels.annotate.annotate import CVAT
from OTLabels.annotate.otc_classes import OtcClass
from OTLabels.annotate.pre_annotate import (
    PreAnnotateImages,
    User,
    collect_images_in,
    move_images,
    select_images,
)
from OTLabels.dataset.generator import (
    SampleType,
    generate_dataset_config,
    generate_image_directories,
)
from OTLabels.helpers.classification import load_classes
from OTLabels.images.import_images import ImportImages
from OTLabels.logger.logger import logger, setup_logger
from OTLabels.tasks.openproject import CreateWorkPackages
from OTLabels.users import load_users

setup_logger()
LOCAL_DATA_PATH: Path = Path("/Users/larsbriem/platomo/data/OTLabels/data_mio_svz")

OTLABELS_PROJECT_ID = 87
CVAT_URL = "https://label.opentrafficcam.org/"
dataset_prefix = "SVZ"
project_name = f"{dataset_prefix}-{SampleType.CORRECT_CLASSIFICATION}"

# data_config = "data/image_data/training_data_svz_all_separate_labels.json"
# data_config = "data/image_data/training_data_svz.json"
class_file = "OTLabels/config/classes_OTC.json"
local_model_file = (
    "/Users/larsbriem/platomo/data/Modelle/"
    "OTCv1-2_yolov8l_mio_batch3_OTC_v0-1-4.mlpackage"
)
gpu_model_file = (
    "/Volumes/platomo data/Produkte/OpenTrafficCam/OTLabels/Modelle"
    "/mioVision/OTCv1-2_yolov8l_mio_batch3_OTC_v0-1-4_fp16.engine"
    # "/OTC/OTCv1-2_yolov8l_OTC_v0-1-4_imgsz800.engine"
)
remote_model_file = (
    "/Volumes/platomo data/Produkte/OpenTrafficCam/OTLabels/Modelle"
    "/mioVision/OTCv1-2_yolov8l_mio_batch3_OTC_v0-1-4_fp16.mlpackage"
)

model_file = remote_model_file
model_file = local_model_file

users = load_users()
classes = load_classes(class_file)
all_classes = classes.keys()
job_size = 5
debug_classes = {OtcClass.MOTORCYCLIST: 1}
upload_classes = classes
# upload_classes = debug_classes
cvat = CVAT(
    url=CVAT_URL,
    project_name=project_name,
    class_file=class_file,
)

date = datetime.now().strftime("%Y-%m-%d")
sample_type = SampleType.CORRECT_CLASSIFICATION
input_path = LOCAL_DATA_PATH
annotation_directory = input_path / f"annotation-{date}"


#
# Start Processing
#
def prepare_images(
    input_path: Path, sample_type: SampleType, annotation_directory: Path
):
    # 1. Alle zu annotierenden Bilder sammlen
    samples_per_class: dict[OtcClass, int] = {
        OtcClass.DELIVERY_VAN: 10,
        OtcClass.TRUCK: 10,
        OtcClass.MOTORCYCLIST: 10,
        OtcClass.PRIVATE_VAN: 10,
        OtcClass.BICYCLIST: 10,
    }
    directories = generate_image_directories(
        base_path=input_path,
        sample_type=sample_type,
        classifications=list(samples_per_class.keys()),
    )
    images = collect_images_in(directories)
    # 2. 100 Bilder auswählen (Mindestabstand zwischen Bildern einhalten, 60 Frames)
    selected_images = select_images(images, samples_per_class)
    logger().info(f"Selected {len(selected_images)} images")
    logger().info(selected_images)
    # 2. Bilder, die zu annotieren sind in einen eigenen Ordner verschieben
    move_images(selected_images, annotation_directory)


prepare_images(input_path, sample_type, annotation_directory)


def generate_user_pairs(users: list[User]) -> list[tuple[User, User]]:
    assignees = [(a, b) for a, b in itertools.product(users, repeat=2) if a != b]
    random.shuffle(assignees)
    logger().info(assignees)
    return assignees


def create_jobs(dataset_name: str, assignees: Iterator[tuple[User, User]]) -> None:
    dataset = fiftyone.load_dataset(dataset_name)
    remaining_dataset = fiftyone.Dataset(f"{dataset_name}_remaining")
    remaining_dataset.add_samples(dataset)
    # iterativ 100 rausnehmen und zuweisen.
    while len(remaining_dataset) > 0:
        for assignee, reviewer in assignees:
            to_assign = remaining_dataset.take(job_size, seed=42)

            to_assign_name = f"{dataset_name}_{assignee.cvat}_{reviewer.cvat}"
            to_assign_name = to_assign_name.replace(" ", "_")
            to_assign_dataset = fiftyone.Dataset(to_assign_name)
            to_assign_dataset.add_samples(to_assign)
            to_assign_dataset.sort_by("site")

            # 3. Task und Job in CVAT anlegen
            tasks = cvat.export_data(
                annotation_key=to_assign_name,
                task_assignee=assignee.cvat,
                job_assignees=[assignee.cvat],
                samples=0,
                task_size=job_size,
                segment_size=job_size,
                exclude_labels=(),
                include_classes=(),
                dataset_name=to_assign_name,
                overwrite_annotation=True,
                keep_samples=False,
            )
            # 4. Issue in OP anlegen
            #   (Enthält Link zu CVAT Task und Job, Bearbeiterhandling in OP)

            CreateWorkPackages(
                project_id=OTLABELS_PROJECT_ID
            ).create_open_project_tasks(tasks, assignee, reviewer)

            remaining_dataset.delete_samples(to_assign)
            if len(remaining_dataset) == 0:
                break


# 1. Assignee und Reviewer definieren
assignees = generate_user_pairs(users)
iterator = itertools.cycle(assignees)
for key, value in upload_classes.items():
    config = generate_dataset_config(
        classifications=[key],
        sample_type=SampleType.CORRECT_CLASSIFICATION,
        base_path=annotation_directory,
    )
    old_key = f"{dataset_prefix}-{key}"
    if fiftyone.dataset_exists(old_key):
        fiftyone.delete_dataset(old_key)
    dataset_name = f"{dataset_prefix}_{key}"

    # 3. Pre-Annotation für diese Bilder durchführen
    PreAnnotateImages(
        config=config,
        class_file=class_file,
        model_file=model_file,
    ).pre_annotate()

    # 1. Daten in fiftyone laden
    importer = ImportImages(
        config=config,
        class_file=class_file,
        otc_pipeline_import=True,
    )
    importer.initial_import(
        import_labels=True,
        launch_app=False,
        dataset_name=dataset_name,
        overwrite=True,
    )

    create_jobs(dataset_name, iterator)
