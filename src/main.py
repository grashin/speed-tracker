import easydict
import json

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys

sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from shapely.geometry import box, Polygon, LineString
import numpy as np



def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs




def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def process_outputs(img, frame_idx, bbox, identities, cars_entering, cars_elapsed_time,
                    area_start, area_end):
    # print('\n')
    # print('\nImg shape', img.shape)
    cv2.polylines(img, np.int32([area_start]), 1, (0, 255, 0), 4)
    cv2.polylines(img, np.int32([area_end]), 1,  (0, 255, 0), 4)
    # print(identities)
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        # box text and bar
        id_car = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id_car)
        # bbox_1 = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(img, (center_x, center_y), 5, (255, 255, 255), -1)
        if cv2.pointPolygonTest(np.array(area_start, np.int32), (int(center_x), int(center_y)), False) > 0:
            cars_entering[id_car] = frame_idx

        label = '{}{:d}'.format("", id_car)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        # cv2.putText(img, label, (x1, y1 +
        #                          t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

        if id_car in cars_entering:
            result = cv2.pointPolygonTest(np.array(area_end, np.int32), (int(center_x), int(center_y)), False)
            if result >= 0:
                elapsed_time = frame_idx - cars_entering[id_car]
                if id_car not in cars_elapsed_time:
                    cars_elapsed_time[id_car] = 1000/60 * elapsed_time
                if id_car in cars_elapsed_time:
                    elapsed_time = cars_elapsed_time[id_car]

                distance = 20  # meters
                a_speed_ms = distance / elapsed_time
                a_speed_kh = int(a_speed_ms * 3600)
                # print('\nCar {} id with speed {} km/h\n'.format(id_car, a_speed_kh))
                # cv2.rectangle(
                #     img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
                cv2.putText(img, str(a_speed_kh), (x1, y1 +
                                         t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)




    return img, cars_entering, cars_elapsed_time


def detect(opt, area_start=None, area_end=None):
    opt.img_size = check_img_size(opt.img_size)
    save_path, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.img_size, opt.evaluate

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays

    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    cars_entering = {}
    cars_elapsed_time = {}


    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_sync()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_sync()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s


            # s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                xywh_bboxs = []
                confs = []

                for *xyxy, conf, cls in det:
                    # to deep sort format
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]

                clss = det[:, 5]
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                # print(outputs)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    im0, cars_entering, cars_elapsed_time = process_outputs(im0, frame_idx, bbox_xyxy, identities=identities,
                                                                            cars_entering=cars_entering,
                                                                            cars_elapsed_time=cars_elapsed_time,
                                                                            area_start=area_start, area_end=area_end)
            else:
                deepsort.increment_ages()

            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    fps_out = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    codec = cv2.VideoWriter_fourcc(*'h264')
                    vid_writer = cv2.VideoWriter(save_path, codec, fps_out, (w, h))
                vid_writer.write(im0)


def process_video(input_video: str, output_video: str, area_start, area_end) -> None:
    deepsort_config = \
        {
            'yolo_weights': 'yolov5/weights/yolov5l.pt',
            'deep_sort_weights': 'deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7',
            # 'output': run_config['out_path'],
            'show_vid': True,
            'save_vid': True,
            'save_txt': False,
            'img_size': 640,
            'evaluate': False,
            'device': '',
            'augment': False,
            'conf_thres': 0.4,
            'fourcc': 'h264',
            'iou_thres': 0.5,
            'classes': None,
            'agnostic_nms': False,
            'config_deepsort': "deep_sort_pytorch/configs/deep_sort.yaml"
            # 'info': False
        }
    deepsort_config['source'] = input_video
    deepsort_config['output'] = output_video
    deepsort_config = easydict.EasyDict(deepsort_config)
    with torch.no_grad():
        detect(deepsort_config, area_start=area_start, area_end=area_end)

process_video('video/road.mov', 'output.mp4', area_start=[(712, 532), (835, 474), (767, 391), (655, 432)], area_end=[(1219, 280), (1276, 247), (1244, 223), (1193, 244)])


