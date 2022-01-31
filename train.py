from yolov5 import train

import config as CONFIG

from OTLabels.preprocessing import get_coco_data
from OTLabels.preprocessing import filter_labels


def main():
    # train.py --weights "yolov5m.pt" --cfg "../OTLabels/models/yolov5m_6cl.yaml"
    # --data "../OTLabels/data/coco_6cl.yaml" --hyp "data/hyp.finetune.yaml"
    # --epochs 150 --batch-size 64 --project "OTLabels" --name "yolo_v5m_6cl_finetune"

    # Get COCO dataset

    get_coco_data.main(
        image_urls=CONFIG.COCO_IMAGE_URLS,
        ann_url=CONFIG.COCO_ANNS_URLS,
        save_path=CONFIG.DATA_DIR,
        force_download=False,
    )

    # Convert to YOLO format

    # Filter COCO dataset
    if CONFIG.FILTER_CLASSES:
        filter_labels.main(
            path=CONFIG.COCO_DIR,
            labels_filter=CONFIG.LABELS_FILTER,
            force_filtering=False,
        )

    """ train.run(
        weights=CONFIG.MODEL_WEIGHTS,
        cfg=CONFIG.MODEL_CFG,
        data=CONFIG.DATA_CONFIG,
        hyp=CONFIG.MODEL_HYP,
        epochs=CONFIG.EPOCHS,
        batch_size=CONFIG.BATCH_SIZE,
        project=CONFIG.PROJECT_NAME,
        name=CONFIG.MODEL_NAME,
    ) """


if __name__ == "__main__":
    main()
