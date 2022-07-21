# limit the number of cpus used by high performance libraries
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys

sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from shapely.geometry import Polygon, box

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams, VID_FORMATS
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

from datetime import datetime
from datetime import timedelta
import xlwt
from xlwt import Workbook

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def get_Contour(img_part):
    img_part_org = img_part
    vectorized = img_part.reshape((-1,3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    attempts=10
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    img_part =np.array( res.reshape((img_part.shape)))

    # Parameters
    blur = 21
    canny_low = 20
    canny_high = 140
    min_area = 0.1
    max_area = 0.95
    dilate_iter = 2
    erode_iter = 2
    mask_color = (0.0,0.0,0.0)

    # Apply Canny Edge Dection
    edges = cv2.Canny(img_part, canny_low, canny_high)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]

    # Get the area of the image as a comparison
    image_area = img_part.shape[0] * img_part.shape[1]

    # calculate max and min areas in terms of pixels
    max_area = max_area * image_area
    min_area = min_area * image_area

    # Set up mask with a matrix of 0's
    mask = np.zeros(edges.shape, dtype=np.uint8)

    # Go through and find relevant contours and apply to mask
    for contour in contour_info:
        # Instead of worrying about all the smaller contours, if the area is smaller than the min, the loop will break
        if contour[1] > min_area and contour[1] < max_area:
            # Add contour to mask
            mask = cv2.fillConvexPoly(mask, contour[0], (255))

    # use dilate, erode, and blur to smooth out the mask
    mask = cv2.dilate(mask, None, iterations=dilate_iter)
    mask = cv2.erode(mask, None, iterations=erode_iter)
    mask = np.where((mask == cv2.GC_BGD), 0, 1).astype(np.uint8)

    contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    maxArea = 0
    contourBox = []
    for cnt in contour:
        rect = cv2.minAreaRect(contour[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        deltaX = np.abs(box[0][0] - box[2][0])
        deltaY = np.abs(box[0][1] - box[2][1])
        area = deltaX * deltaY
        if maxArea < area:
            maxArea = area
            contourBox = [box]
    # cv2.drawContours(img_part_org, contourBox, 0, (0, 0, 255), 2)
    return contourBox
def get_spot_info(bboxes,intersectionThreshold, img, im0):
    parking_spots = getSpotsInfo(img, im0)
    # parking_spots = [(Polygon([(180, 70), (140, 70), (140, 130), (210, 125)]), "A1"),
    #                  (Polygon([(140, 130), (210, 125), (240, 200), (140, 210)]), "A2"),
    #                  (Polygon([(240, 200), (140, 210), (160, 400), (300, 400)]), "A3"),
    #                  (Polygon([(289, 70), (335, 54), (388, 91), (326, 111)]), "B1"),
    #                  (Polygon([(388, 91), (326, 111), (388, 177), (458, 147)]), "B2"),
    #                  (Polygon([(553, 337), (613, 269), (458, 147), (388, 177)]), "B3")]
    parkingSpotIndex = -1
    max_percentage = 0
    for spot_index in range(len(parking_spots)):
        startX = int(min([bboxes[0],bboxes[2]]))
        endX = int(max([bboxes[0],bboxes[2]]))
        startY = int(min([bboxes[1],bboxes[3]]))
        endY = int(max([bboxes[1],bboxes[3]]))
        bb = img[startY:endY, startX:endX]
        # contour = get_Contour(bb)
        poly = Polygon([(bboxes[0], bboxes[1]), (bboxes[0], bboxes[3]), (bboxes[2], bboxes[3]),(bboxes[2], bboxes[1])])
        # if len(contour) != 0:
        #     # cv2.drawContours(img_part_org, contour, -1, (0, 255, 0))hjj
        #     poly_draw = np.array([[contour[0][j][0] + startX, contour[0][j][1] + startY] for j in range(len(contour[0]))])
        #     img = cv2.polylines(img=img, pts=np.int32([list(poly_draw)]), isClosed=True, color=(255,127,80), thickness=1)
        #     poly = Polygon([(contour[0][j][0] + startX, contour[0][j][1] + startY) for j in range(len(contour[0]))])

        p2 = parking_spots[spot_index][0]
        p3 = poly.intersection(p2)
        # print(p3) # result: POLYGON ((0.5 0.5, 1 1, 1 0, 0.5 0.5))
        intersectionPercentage = (p3.area / poly.area) * 100
        if max_percentage < intersectionPercentage and intersectionPercentage > intersectionThreshold:
            max_percentage = intersectionPercentage
            parkingSpotIndex = spot_index
    if parkingSpotIndex == -1:
        return "", 0
    return parking_spots[parkingSpotIndex][1], int(max_percentage)
def drawBoundaries(frame):
    parking_spotB_1 = np.array([[289, 70], [335, 54], [388, 91], [326, 111]])
    parking_spotB_2 = np.array([[388, 91], [326, 111], [388, 177], [458, 147]])
    parking_spotB_3 = np.array([[553, 337], [613, 269], [458, 147], [388, 177]])
    frame_with_parking_lines = cv2.polylines(img=frame, pts=np.int32([list(parking_spotB_1)]), isClosed=True,
                                             color=(255, 120, 255), thickness=3)
    frame_with_parking_lines = cv2.polylines(img=frame, pts=np.int32([list(parking_spotB_2)]), isClosed=True,
                                             color=(255, 120, 255), thickness=3)
    frame_with_parking_lines = cv2.polylines(img=frame, pts=np.int32([list(parking_spotB_3)]), isClosed=True,
                                             color=(255, 120, 255), thickness=3)

    parking_spotA_1 = np.array([[180, 70], [140, 70], [140, 130], [210, 125]])
    parking_spotA_2 = np.array([[140, 130], [210, 125], [240, 200], [140, 210]])
    parking_spotA_3 = np.array([[240, 200], [140, 210], [160, 400], [300, 400]])
    frame_with_parking_lines = cv2.polylines(img=frame, pts=np.int32([list(parking_spotA_1)]), isClosed=True,
                                             color=(255, 120, 255), thickness=3)
    frame_with_parking_lines = cv2.polylines(img=frame, pts=np.int32([list(parking_spotA_2)]), isClosed=True,
                                             color=(255, 120, 255), thickness=3)
    frame_with_parking_lines = cv2.polylines(img=frame, pts=np.int32([list(parking_spotA_3)]), isClosed=True,
                                             color=(255, 120, 255), thickness=3)
    return frame_with_parking_lines
def get_time(seconds):
    # video_start_time = '31/7/2021 17:51:37'
    video_start_time = '31/7/2021 18:11:19'
    date_format_str = '%d/%m/%Y %H:%M:%S'
    given_time = datetime.strptime(video_start_time, date_format_str)
    final_time = given_time + timedelta(seconds=seconds)
    return final_time

def write_to_excel(records, sourceFile):
    wb = Workbook()
    # add_sheet is used to create sheet.
    sheet1 = wb.add_sheet(sourceFile)

    sheet1.write(0, 1, 'car ID')
    sheet1.write(0, 2, 'Slot ID')
    sheet1.write(0, 3, 'Arrival time')
    sheet1.write(0, 4, 'Departure time')
    sheet1.write(0, 5, 'Duration')
    sheet1.write(0, 6, 'Intersection')
    # records = {3.0: {'A2': ([datetime.datetime(2021, 7, 31, 17, 52, 37), datetime.datetime(2021, 7, 31, 18, 2, 15)], [92, 97])},
    #            8.0: {'A1': ([datetime.datetime(2021, 7, 31, 17, 52, 37), datetime.datetime(2021, 7, 31, 18, 2, 15)], [94, 98])},
    #            19.0: {'B2': ([datetime.datetime(2021, 7, 31, 17, 52, 43), datetime.datetime(2021, 7, 31, 18, 2, 15)], [51, 87])},
    #            24.0: {'B3': ([datetime.datetime(2021, 7, 31, 17, 52, 46), datetime.datetime(2021, 7, 31, 18, 2, 15)], [75, 82])},
    #            46.0: {'B1': ([datetime.datetime(2021, 7, 31, 17, 52, 59), datetime.datetime(2021, 7, 31, 18, 2, 15)], [72, 87])},
    #            735.0: {'A3': ([datetime.datetime(2021, 7, 31, 18, 1, 33), datetime.datetime(2021, 7, 31, 18, 2, 15)], [38, 39])}}
    #records[id] = {spot_name: ([curr_time, curr_time], [percent, percent])}
    record_idx = 1
    for car_ID in records:
        sheet1.write(record_idx, 1, car_ID)
        spot_idx = 0
        for spot_ID in records[car_ID]:
            spot_data = records[car_ID][spot_ID]
            duration = (spot_data[0][1] - spot_data[0][0])
            sheet1.write(record_idx + spot_idx, 2, spot_ID)
            sheet1.write(record_idx + spot_idx, 3, spot_data[0][0].strftime('%d/%m/%Y %H:%M:%S'))
            sheet1.write(record_idx + spot_idx, 4, spot_data[0][1].strftime('%d/%m/%Y %H:%M:%S'))
            sheet1.write(record_idx + spot_idx, 5, str(duration))
            sheet1.write(record_idx + spot_idx, 6, int((spot_data[1][0] + spot_data[1][1]) / 2))
            spot_idx += 1
        record_idx += spot_idx

    if not os.path.exists('xls'):
        os.makedirs('xls')
    wb.save('xls/'+sourceFile+'.xls')

records = {}
frame_num = 0
def update_records(id, spot_name, percent):
    global frame_num
    curr_time = get_time(frame_num)
    car_info = records.get(id)
    if car_info != None:
        car_spot_info = car_info.get(spot_name)
        if car_spot_info != None:
            recorded_time = car_spot_info[0]
            recorded_time[1] = curr_time
            recorded_percent = car_spot_info[1]
            if recorded_percent[0] > percent:
                recorded_percent[0] = percent
            elif recorded_percent[1] < percent:
                recorded_percent[1] = percent
        else:
            records[id][spot_name] = ([curr_time, curr_time], [percent, percent])
    else:
        records[id] = {spot_name: ([curr_time, curr_time], [percent, percent])}

def detectOccupancy(im, im0):
    modelPath = "yolov5/weights/yoloOccupancy.pt"
    device = select_device(opt.device)
    model = DetectMultiBackend(modelPath, device=device, dnn=True)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = (640, 640)
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    pred = model(im, augment=opt.augment, visualize=False)
    pred = non_max_suppression(pred)
    boxes = []
    count = 0
    for i, det in enumerate(pred):
        count += 1
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            xywhs = xyxy2xywh(det[:, 0:4])
            LOGGER.info(f'class : {det[:, -1].unique()}, points : {xywhs}')

            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = (f'{names[c]} {conf:.2f}')
                annotator = Annotator(im0, line_width=2, example=str(names))
                annotator.box_label(xyxy, label, color=colors(c, True))
                LOGGER.info(f'points : {xyxy[0].item()}')
                p1x = xyxy[0].item()
                p1y = xyxy[1].item()
                p2x = xyxy[2].item()
                p2y = xyxy[3].item()
                boxes.append((box(p1x, p1y, p2x, p2y),"x"))
        # im0 = annotator.result()
        # cv2.imwrite(str(count)+".jpg", im0)
    # raise Exception(boxes)
    LOGGER.info(boxes)
    return boxes
def getSpotsInfo(image, im0):
    pred = detectOccupancy(image, im0)
    return pred

def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, \
    project, exist_ok, update, save_crop, intersectionThreshold, initializationTime= \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.exist_ok, opt.update, opt.save_crop, opt.intersectionThreshold, \
        opt.initTime
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    if type(yolo_model) is str:  # single yolo model
        exp_name = yolo_model.split(".")[0]
    elif type(yolo_model) is list and len(yolo_model) == 1:  # single models after --yolo_model
        exp_name = yolo_model[0].split(".")[0]
    else:  # multiple models after --yolo_model
        exp_name = "ensemble"
    exp_name = exp_name + "_" + deep_sort_model.split('/')[-1].split('.')[0]
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run if project name exists
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    cfg.DEEPSORT.N_INIT = initializationTime*60

    # Create as many trackers as there are video sources
    deepsort_list = []
    for i in range(nr_sources):
        deepsort_list.append(
            DeepSort(
                deep_sort_model,
                device,
                max_dist=cfg.DEEPSORT.MAX_DIST,
                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
            )
        )
    outputs = [None] * nr_sources

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        global frame_num
        frame_num += 1
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if opt.visualize else False
        pred = model(im, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs[i] = deepsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        # LOGGER.info(f'bbox : {bboxes}, id : {id}')

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))
                        spot_name, percentage = get_spot_info(bboxes, intersectionThreshold, im, im0)
                        if spot_name != "":
                            update_records(id, spot_name, percentage)
                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = f'{id} {names[c]} No Spot'
                            if percentage != 0:
                                label = f'{id} {names[c]} Spot {spot_name} {percentage}%'
                            annotator.box_label(bboxes, label, color=colors(c, True))
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[
                                    c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort_list[i].increment_ages()
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            im0 = drawBoundaries(im0)
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_model)  # update model (to fix SourceChangeWarning)
    LOGGER.info(records)
    write_to_excel(records,source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_ibn_x1_0_MSMT17')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--intersectionThreshold', type=float, default=30.0, help='minimum intersection percentage')
    parser.add_argument('--initTime', type=int, default=1, help='the time to decide that the car is parking')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
