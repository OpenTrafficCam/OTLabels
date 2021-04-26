call .\venv\Scripts\activate

@REM yolov5m retrain coco 6cl
cd yolov5
python .\train.py --resume
