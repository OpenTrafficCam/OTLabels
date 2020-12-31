#%% Setup training env for yolo v5
import subprocess
import logging

path_yolo = 'D:/OTLabels/detectors/yolov5/'
name_run = 'test'

path_train = path_yolo
path_yaml = path_yolo + 'data/'
path_weights = path_yolo
path_output = path_yolo + 'runs/train/'

img_size = 640
batch_size = 6
epochs = 30

train_cmd = ('python ' + path_yolo + 'train.py '+ 
             '--img ' + img_size + ' '+
             '--batch ' + batch_size + ' '+
             '--epochs ' + epochs + ' '+
             '--data ' + path_yaml + 'traffic_data.yaml '+
             '--hyp ' + path_yaml + 'hyp.scratch.yaml '+
             '--weights ' + path_weights + 'yolov5x.pt '+
             '--project ' + path_output + ' '+
             '--name ' + name_run + ' '+
             '--nosave '+
             '--cache '+
             '--device 0')

def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

#%% Execute training
for path in execute(train_cmd):
    print(path, end="")






