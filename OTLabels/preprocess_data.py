"""Preprocess image data for annotation in CVAT"""

# from annotate.pre_annotate import PreAnnotateImages

from OTLabels.annotate.annotate import CVAT
from OTLabels.images.import_images import ImportImages

data_config = "data/image_data/training_data_svz.json"
class_file = "OTLabels/config/classes_OTC.json"
model_file = (
    "/Users/larsbriem/platomo/data/Modelle/"
    "OTCv1-2_yolov8l_mio_batch3_OTC_v0-1-4.mlpackage"
)
# PreAnnotateImages(
#     config_file=data_config,
#     class_file=class_file,
#     model_file=model_file,
# ).pre_annotate()

importer = ImportImages(
    config_file=data_config,
    class_file=class_file,
    otc_pipeline_import=True,
)
# imp.delete_dataset("OTLabels_no_bicycles_preannotated")
dataset_name = "SVZ_Img"
importer.initial_import(
    import_labels=True,
    launch_app=True,
    dataset_name=dataset_name,
    overwrite=True,
)

cvat = CVAT(
    url="https://label.opentrafficcam.org/",
    project_name="SVZ-Test",
    class_file=class_file,
)
for i in range(0, 1):
    cvat.export_data(
        annotation_key=f"SVZ_samples_{i}",
        samples=1000,
        segment_size=100,
        exclude_labels=(),  # ("bicyclist", "motorcyclist"),
        include_classes=(),  # ("pedestrian", "truck", "bus"),
        dataset_name=dataset_name,
        overwrite_annotation=True,
        keep_samples=False,
    )
# cvat.import_data(anno_key="manual_samples", launch_app=True)
