"""Preprocess image data for annotation in CVAT"""

from annotate.pre_annotate import PreAnnotateImages

from OTLabels.annotate.annotate import CVAT
from OTLabels.images.import_images import ImportImages

PreAnnotateImages(
    config_file="data/image_data/training_data.json",
    class_file="OTLabels/config/classes_COCO.json",
    model_file="yolov8m.pt",
).pre_annotate()

ImportImages(
    config_file="data/image_data/training_data.json",
    class_file="OTLabels/config/classes_COCO.json",
).initial_import(
    import_labels=True,
    launch_app=True,
    overwrite=True,
)

cvat = CVAT(
    url="https://label.opentrafficcam.org/",
    project_name="test_fiftyone",
    class_file="OTLabels/config/classes_OTC.json",
)

cvat.export_data(anno_key="manual_samples")
cvat.import_data(anno_key="manual_samples", launch_app=True)
