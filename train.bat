call .\venv\Scripts\activate

@REM yolov5m retrain coco 6cl
cd yolov5
python .\train.py --weights "yolov5m.pt" --cfg "../OTLabels/models/yolov5m_6cl.yaml" --data "../OTLabels/data/coco_6cl.yaml" --epochs 150 --batch-size 64 --project "OTLabels" --name "yolo_v5m_6cl"
