"""Preprocess image data for annotation in CVAT"""

# from annotate.pre_annotate import PreAnnotateImages

from OTLabels.annotate.annotate import CVAT

# from OTLabels.images.import_images import ImportImages

# PreAnnotateImages(
#     config_file="data/image_data/training_data.json",
#     class_file="OTLabels/config/classes_COCO.json",
#     model_file="yolov8m.pt",
# ).pre_annotate()

# imp = ImportImages(
#     config_file="data/image_data/training_data.json",
#     class_file="OTLabels/config/classes_COCO.json",
# )
# imp.delete_dataset("OTLabels_no_bicycles_preannotated")
# imp.initial_import(
#     import_labels=True,
#     launch_app=True,
#     name="OTLabels",
#     overwrite=True,
# )

cvat = CVAT(
    url="https://label.opentrafficcam.org/",
    project_name="OTLabels",
    class_file="OTLabels/config/classes_OTC.json",
)
cvat.export_data(
    anno_key="first_samples",
    samples=1000,
    segment_size=100,
    exclude_labels=("bicyclist", "motorcyclist"),
    job_reviewers=["martin", "michael"],
    include_classes=("pedestrian", "truck", "bus", "motorcyclist"),
    overwrite=True,
)
# cvat.import_data(anno_key="manual_samples", launch_app=True)
