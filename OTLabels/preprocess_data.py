"""Preprocess image data for annotation in CVAT"""

import json

from OTLabels.annotate.annotate import CVAT
from OTLabels.dataset.generator import generate_dataset_config
from OTLabels.images.import_images import ImportImages

# from annotate.pre_annotate import PreAnnotateImages


data_config = "data/image_data/training_data_svz_all_separate_labels.json"
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
classes = {}
with open(class_file) as json_file:
    classes = json.load(json_file)
all_classes = classes.keys()

dataset_prefix = "SVZ_Img"

# PreAnnotateImages(
#     config_file=data_config,
#     class_file=class_file,
#     model_file=model_file,
# ).pre_annotate()
upload_classes = classes
manual_classes = {
    "truck_with_semitrailer": 16,
    "other": 17,
}
upload_classes = classes
cvat = CVAT(
    url="https://label.opentrafficcam.org/",
    project_name="SVZ-different-class",
    class_file=class_file,
)

for key, value in upload_classes.items():
    config = generate_dataset_config([key])
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

    cvat.export_data(
        annotation_key=f"SVZ_samples_{key}",
        samples=0,
        segment_size=100,
        exclude_labels=(),
        include_classes=(),
        dataset_name=dataset_name,
        overwrite_annotation=True,
        keep_samples=False,
    )
