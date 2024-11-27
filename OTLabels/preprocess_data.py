"""Preprocess image data for annotation in CVAT"""

from datetime import datetime
from pathlib import Path

from OTLabels.annotate.annotate import CVAT
from OTLabels.annotate.otc_classes import OtcClass
from OTLabels.annotate.pre_annotate import collect_images_in, move_images, select_images
from OTLabels.dataset.generator import (
    SampleType,
    generate_dataset_config,
    generate_image_directories,
)
from OTLabels.helpers.classification import load_classes
from OTLabels.images.import_images import ImportImages

LOCAL_DATA_PATH: Path = Path("/Users/larsbriem/platomo/data/OTLabels/data_mio_svz")

CVAT_URL = "https://label.opentrafficcam.org/"
dataset_prefix = "SVZ"
project_name = f"{dataset_prefix}-{SampleType.CORRECT_CLASSIFICATION}"

data_config = "data/image_data/training_data_svz_all_separate_labels.json"
data_config = "data/image_data/training_data_svz.json"
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

# 1. Alle zu annotierenden Bilder sammlen
samples_per_class: dict[OtcClass, int] = {
    OtcClass.DELIVERY_VAN: 500,
    OtcClass.TRUCK: 500,
    OtcClass.MOTORCYCLIST: 500,
    OtcClass.PRIVATE_VAN: 500,
    OtcClass.BICYCLIST: 500,
}
sample_type = SampleType.CORRECT_CLASSIFICATION
input_path = LOCAL_DATA_PATH
directories = generate_image_directories(
    base_path=input_path,
    sample_type=sample_type,
    classifications=list(samples_per_class.keys()),
)
images = collect_images_in(directories)
# 2. Bilder, die zu annotieren sind in einen eigenen Ordner verschieben
date = datetime.now().strftime("%Y-%m-%d")
output_path = LOCAL_DATA_PATH / f"annotation-{date}" / sample_type
selected_images = select_images(images, samples_per_class)
move_images(selected_images, output_path)
# 1. Assignee und Reviewer definieren
# 2. 100 Bilder auswählen (Mindestabstand zwischen Bildern einhalten, 60 Frames)
# 3. Pre-Annotation für diese Bilder durchführen
# 3. Task und Job in CVAT anlegen
# 4. Issue in OP anlegen (Enthält Link zu CVAT Task und Job, Bearbeiterhandling in OP)
#


# PreAnnotateImages(
#     config_file=data_config,
#     class_file=class_file,
#     model_file=model_file,
# ).pre_annotate()

debug_classes = {OtcClass.BICYCLIST: 1}
upload_classes = classes
upload_classes = debug_classes
cvat = CVAT(
    url=CVAT_URL,
    project_name=project_name,
    class_file=class_file,
)

for key, value in upload_classes.items():
    config = generate_dataset_config(
        classifications=[key],
        sample_type=SampleType.CORRECT_CLASSIFICATION,
        base_path=LOCAL_DATA_PATH,
    )
    dataset_name = f"{dataset_prefix}_{key}"
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

    tasks = cvat.export_data(
        annotation_key=dataset_name,
        samples=0,
        task_size=100,
        segment_size=100,
        exclude_labels=(),
        include_classes=(),
        dataset_name=dataset_name,
        overwrite_annotation=True,
        keep_samples=False,
    )

    print(tasks)
