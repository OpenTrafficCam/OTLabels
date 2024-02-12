"""Preprocess image data for annotation in CVAT"""

from annotate.pre_annotate import PreAnnotateImages

from OTLabels.annotate.annotate import CVAT
from OTLabels.images.import_images import ImportImages

PreAnnotateImages(
    config_file="data/image_data/training_data_mioVision_SH.json",
    class_file="OTLabels/config/classes_OTC.json",
    model_file="data/prototype/models/OTCv1-2_yolo_v8s_mio.pt",
).pre_annotate()


importer = ImportImages(
    config_file="data/image_data/training_data_mioVision_SH.json",
    class_file="OTLabels/config/classes_OTC.json",
)
# imp.delete_dataset("OTLabels_no_bicycles_preannotated")
importer.initial_import(
    import_labels=True,
    launch_app=True,
    name="mioVision_SH",
    overwrite=True,
)

cvat = CVAT(
    url="https://label.opentrafficcam.org/",
    project_name="OTLabels",
    class_file="OTLabels/config/classes_OTC.json",
)
# for i in range(0, 5):
cvat.export_data(
    annotation_key="SH_samples_MioVision",
    # samples=1000,
    segment_size=100,
    # exclude_labels=("bicyclist", "motorcyclist"),
    # include_classes=("pedestrian", "truck", "bus"),
    dataset_name="mioVision_SH",
    overwrite_annotation=True,
)
# cvat.import_data(anno_key="manual_samples", launch_app=True)
