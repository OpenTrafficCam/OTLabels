from yolov5 import train

import config as CONFIG

from OTLabels.preprocessing import get_coco_data
from OTLabels.preprocessing import filter_labels
from OTLabels.preprocessing import cvat_to_yolo


def main():
    # Get COCO dataset
    get_coco_data.main(
        image_urls=CONFIG.COCO_IMAGE_URLS,
        ann_url=CONFIG.COCO_ANNS_URLS,
        save_path=CONFIG.DATA_DIR,
        force_download=CONFIG.FORCE_DOWNLOAD_COCO,
    )

    # Filter COCO dataset
    if CONFIG.FILTER_CLASSES:
        filter_labels.main(
            path=CONFIG.COCO_DIR,
            labels_filter=CONFIG.LABELS_CVAT,
            force_filtering=CONFIG.FORCE_FILTERING_LABELS,
        )

    if CONFIG.USE_CUSTOM_DATSETS:
        cvat_to_yolo.main(
            dest_path=CONFIG.CUSTOM_DIR,
            cvat_path=CONFIG.CUSTOM_DATASETS,
            labels_cvat_path=CONFIG.LABELS_CVAT,
            coco_ann_file_path=CONFIG.COCO_JSON_FILE,
            new_dataset_name=CONFIG.CUSTOM_DATASET_NAME,
        )

    train.run(
        weights=CONFIG.MODEL_WEIGHTS,
        cfg=CONFIG.MODEL_CFG,
        data=CONFIG.DATA_CONFIG,
        hyp=CONFIG.MODEL_HYP,
        epochs=CONFIG.EPOCHS,
        batch_size=CONFIG.BATCH_SIZE,
        project=CONFIG.PROJECT_NAME,
        name=CONFIG.MODEL_NAME,
    )


if __name__ == "__main__":
    main()
