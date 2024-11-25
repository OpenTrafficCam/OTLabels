"""Preprocess image data for annotation in CVAT"""

import json
from pathlib import Path

from OTLabels.annotate.annotate import CVAT
from OTLabels.dataset.generator import generate_dataset_config
from OTLabels.images.import_images import ImportImages

LOCAL_DATA_PATH: Path = Path("/Users/larsbriem/platomo/data/OTLabels/data_mio_svz")
PROJECT_TYPE = "correct_classification"

CVAT_URL = "https://label.opentrafficcam.org/"
dataset_prefix = "SVZ"
project_name = f"{dataset_prefix}-{PROJECT_TYPE}"

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

# TODO
# 1. Assignee und Reviewer definieren
# 2. 100 Bilder auswählen (Mindestabstand zwischen Bildern einhalten, 60 Frames)
# 3. Pre-Annotation für diese Bilder durchführen
# 3. Task und Job in CVAT anlegen
# 4. Issue in OP anlegen (Enthält Link zu CVAT Task und Job)
#


model_file = remote_model_file
classes = {}
with open(class_file) as json_file:
    classes = json.load(json_file)
all_classes = classes.keys()

# PreAnnotateImages(
#     config_file=data_config,
#     class_file=class_file,
#     model_file=model_file,
# ).pre_annotate()

debug_classes = {"bicyclist": 1}
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
        project_type=PROJECT_TYPE,
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
