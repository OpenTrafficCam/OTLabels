from yolov5 import train

import config as CONFIG


def main():
    train.run(
        weights=CONFIG.MODEL_WEIGHTS,
        cfg=CONFIG.MODEL_CFG,
        data=CONFIG.DATA_CONFIG,
        hyp=CONFIG.MODEL_HYP,
        epochs=CONFIG.EPOCHS,
        batch_size=CONFIG.BATCH_SIZE,
        project=CONFIG.PROJECT_NAME,
        name=CONFIG.MODEL_NAME,
        resume=CONFIG.RESUME_TRAINING,
    )


if __name__ == "__main__":
    main()
