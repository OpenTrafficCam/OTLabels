call .\venv\Scripts\activate

@REM yolov5m retrain coco 6cl
cd yolov5
@REM python .\train.py --weights "yolov5m.pt" --cfg "../OTLabels/models/yolov5m_6cl.yaml" --data "../OTLabels/data/coco_6cl.yaml" --epochs 150 --batch-size 64 --project "OTLabels" --name "yolo_v5m_6cl"
python .\train.py --weights "yolov5m.pt" --cfg "../OTLabels/models/yolov5m_6cl.yaml" --data "../OTLabels/data/coco_6cl.yaml" --hyp "data/hyp.finetune.yaml" --epochs 150 --batch-size 64 --project "OTLabels" --name "yolo_v5m_6cl_finetune"
