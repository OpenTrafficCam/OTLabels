"""Preprocess image data for annotation in CVAT"""

# from annotate.pre_annotate import PreAnnotateImages

from OTLabels.annotate.annotate import CVAT
from OTLabels.images.import_images import ImportImages

# PreAnnotateImages(
#     config_file="data/image_data/training_data.json",
#     class_file="OTLabels/config/classes_COCO.json",
#     model_file="yolov8m.pt",
# ).pre_annotate()


imp = ImportImages(
    config_file="data/image_data/training_data_T30.json",
    class_file="OTLabels/config/classes_COCO.json",
)
# imp.delete_dataset("OTLabels_no_bicycles_preannotated")
imp.initial_import(
    import_labels=True,
    launch_app=True,
    name="T30_img",
    overwrite=True,
)

cvat = CVAT(
    url="https://label.opentrafficcam.org/",
    project_name="OTLabels",
    class_file="OTLabels/config/classes_OTC.json",
)
for i in range(0, 5):
    cvat.export_data(
        anno_key=f"T30_samples_{i}",
        samples=1000,
        segment_size=100,
        exclude_labels=("bicyclist", "motorcyclist"),
        include_classes=("pedestrian", "truck", "bus"),
        dataset_name="T30_img",
        overwrite_annotation=True,
    )
    i += 1
# cvat.import_data(anno_key="manual_samples", launch_app=True)
