from yolov5 import train

import config as CONFIG
from config import load_config

from OTLabels.preprocessing import get_coco_data
from OTLabels.preprocessing import filter_labels
from OTLabels.preprocessing import cvat_to_yolo


def main(config_path):
    config = load_config(config_path)

    # Get COCO dataset
    get_coco_data.main(
        image_urls=config["coco_img_urls"],
        ann_url=config["coco_anns_urls"],
        save_path=config["data_dir"],
        force_download=config["force_download_coco"],
    )

    # Filter COCO dataset
    if config["filter_classes"]:
        filter_labels.main(
            path=config["coco_dir"],
            labels_filter=config["labels_cvat"],
            force_filtering=config["force_filtering_labels"],
        )

    if config["use_custom_datasets"]:
        cvat_to_yolo.main(
            dest_path=config["custom_dir"],
            cvat_path=config["custom_datasets"],
            labels_cvat_path=config["labels_cvat"],
            coco_ann_file_path=config["coco_json_file"],
            new_dataset_name=config["custom_dataset_name"],
        )

    train.run(
        weights=config["model_weights"],
        cfg=config["model_cfg"],
        data=config["data_config"],
        hyp=config["model_hyp"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        project=config["project_name"],
        name=config["model_name"],
    )


if __name__ == "__main__":
    config_path = CONFIG.MODEL_TRAINING_CONFIG_PATH
    main(config_path)
