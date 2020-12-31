#%% Setup test env for yolo v5
import subprocess
import logging

path_yolo = 'D:/OTLabels/detectors/yolov5/'
name_run = 'test'

path_train = path_yolo
path_source = path_yolo + 'data/detection'
path_weights = path_yolo
path_output = path_yolo + 'runs/train/'

img_size = 640
conf = 0.5
iou = 0.45

train_cmd = ('python ' + path_yolo + 'test.py ' 
             '--source ' + path_source + ' '+
             '--weights ' + path_weights + 'yolov5x.pt '+
             '--project ' + path_output + ' '+
             '--name ' + name_run + ' '+
             '--img-size ' + img_size + ' '+ 
             '--conf-thres'  + conf + ' '+
             '--iou-thres ' + iou + ' '+
             '--device 0')

def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

#%% Execute test
for path in execute(train_cmd):
    print(path, end="")
