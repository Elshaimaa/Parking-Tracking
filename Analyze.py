
# %pip install -qr requirements.txt  # install dependencies
import os
import torch
import argparse
from IPython.display import Image, clear_output  # to display images

import cv2
import sys
from io import BytesIO
from PIL import Image, ImageDraw
from copy import copy, deepcopy

def preprocess(intersectionThreshold, initTime):
    #download videos from the drive
    os.system("pip install -qr requirements.txt")
    os.system("pip install -qr 'yolov7/requirements.txt'")
    os.system("gdown --id 11JywddUoylzK6Km1IUeMdJi1jCP7VrDE")
    os.system("unrar x \"1st location-7th st-P bandar.rar\"")

    #rename and move videos
    source = '1st location-7th st-P bandar/10.10.9.8/2021-07-31/'
    destination = ''

    allfiles = os.listdir(source)
    counter = len(allfiles)
    for f in allfiles:
        os.rename(source + f, destination + str(counter)+'.mp4')
        counter -= 1

    counter = len(allfiles)
    #setup gpu if exist
    print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

    #download model weights
    if not os.path.exists('yolov5/weights'):
        os.makedirs('yolov5/weights')
    # # get yolov5m model trained on the crowd-human dataset
    os.system("wget -nc https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5m.pt -O yolov5/weights/yolov5m.pt")

    #take frame from every second
    for c in range(1,counter):
        src_dir = str(c)+ ".mp4"
        dst_dir = str(c)+ ".avi"

        video_cap = cv2.VideoCapture(src_dir)

        fps = video_cap.get(cv2.CAP_PROP_FPS)
        print(fps)
        size = (int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        video_writer = cv2.VideoWriter(dst_dir, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

        success = True
        myFrameNumber = 0
        last_size = 0
        while success:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES,myFrameNumber)
            success, frame = video_cap.read()
            if not success:
              break
            myFrameNumber += fps
            video_writer.write(frame)
        cv2.destroyAllWindows()
        video_cap.release()
        os.system(f"python track.py --yolo_model yolov5/weights/yolov5m.pt --classes 2 5 7 --source {dst_dir} --intersectionThreshold {intersectionThreshold} --initTime {initTime} --save-vid")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--intersectionThreshold', type=float, default=30.0, help='minimum intersection percentage')
    parser.add_argument('--initTime', type=int, default=1, help='the time to decide that the car is parking')
    opt = parser.parse_args()
    preprocess(opt.intersectionThreshold, opt.initTime)
