"""Preprocess image data for annotation in CVAT"""

import itertools
from datetime import datetime
from pathlib import Path

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

setup_logger()
LOCAL_DATA_PATH: Path = Path("/Users/larsbriem/platomo/data/OTLabels/data_mio_svz")

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

classes = load_classes(class_file)
all_classes = classes.keys()
job_size = 5
debug_classes = {OtcClass.BICYCLIST: 1}
upload_classes = classes
upload_classes = debug_classes
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


# prepare_images(input_path, sample_type, annotation_directory)

for key, value in upload_classes.items():
    config = generate_dataset_config(
        classifications=[key],
        sample_type=SampleType.CORRECT_CLASSIFICATION,
        base_path=annotation_directory,
    )
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

    # 1. Assignee und Reviewer definieren
    users = [
        User(open_project="Lars Briem", cvat="Lars"),
        User(open_project="Randy Seng", cvat="Randy"),
    ]
    assignees = [(a, b) for a, b in itertools.product(users, repeat=2) if a != b]
    logger().info(assignees)
    dataset = fiftyone.load_dataset(dataset_name)
    # Beispiel: Splitte das Dataset in zwei zufällige Datasets
    # 70% für Training, 30% für Testen
    # iterativ 100 rausnehmen und zuweisen.
    train_view = dataset.take(0.7 * len(dataset), random_seed=42)
    test_view = dataset.exclude(train_view)

    # Erstelle das Trainings-Dataset und füge die Sichten hinzu
    train_dataset = fiftyone.Dataset("train_dataset")
    train_dataset.add_samples(train_view)

    # Erstelle das Test-Dataset und füge die Sichten hinzu
    test_dataset = fiftyone.Dataset("test_dataset")
    test_dataset.add_samples(test_view)
    # 3. Task und Job in CVAT anlegen
    tasks = cvat.export_data(
        annotation_key=dataset_name,
        task_assignee="",
        job_assignees=[""],
        samples=0,
        task_size=job_size,
        segment_size=job_size,
        exclude_labels=(),
        include_classes=(),
        dataset_name=dataset_name,
        overwrite_annotation=True,
        keep_samples=False,
    )
    # 4. Issue in OP anlegen
    #   (Enthält Link zu CVAT Task und Job, Bearbeiterhandling in OP)

    print(tasks)
