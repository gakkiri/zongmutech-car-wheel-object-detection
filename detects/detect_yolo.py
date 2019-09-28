from __future__ import division

from yolov3.models import *
from yolov3.yolo_utils.utils import *
from yolov3.yolo_utils.datasets import *
from soft_nms import py_cpu_softnms

import os
import sys
import time
import datetime
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, default="samples/", help="path to dataset")
parser.add_argument("--model_def", type=str, default="yolov3/config/yolov3-custom.cfg", help="path to model definition file")
parser.add_argument("--weights_path", type=str, default="yolov3/weights/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="yolov3/classes.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.3, help="iou thresshold for non-maximum suppression")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=608, help="size of each image dimension")
parser.add_argument("--checkpoint_model", type=str, default="checkpoints/yolov3_ckpt_140.pth",
                    help="path to checkpoint model")
opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Load YOLOv3...')
model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
model.load_state_dict(torch.load(opt.checkpoint_model))
model.eval()  # Set in evaluation mode
print(f'YOLOv3 Done\n')


def detect(img, save=False, vis=False, flip=False):
    classes = load_classes(opt.class_path)  # Extracts class labels from file
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    orig_img = img.copy()

    img = transforms.ToTensor()(img)
    img, _ = pad_to_square(img, 0)
    img = resize(img, opt.img_size)
    img = img.unsqueeze(0)

    # print("\nPerforming object detection:")
    prev_time = time.time()

    # Configure input
    input_imgs = Variable(img.type(Tensor))

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)  # (center x, center y, width, height, ...)
        # detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

    if flip:
        flip_img = orig_img.transpose(Image.FLIP_LEFT_RIGHT)
        flip_img = transforms.ToTensor()(flip_img)
        flip_img, _ = pad_to_square(flip_img, 0)
        flip_img = resize(flip_img, opt.img_size)
        flip_img = flip_img.unsqueeze(0)

        input_imgs = Variable(flip_img.type(Tensor))

        with torch.no_grad():
            flip_detections = model(input_imgs)

        flip_anno = flip_detections[..., :4].squeeze()  # (cx, cy, w, h)
        flip_anno[:, 0] = opt.img_size - flip_anno[:, 0]
        flip_detections[..., :4] = flip_anno.unsqueeze(0)

        detections = torch.cat([detections, flip_detections], dim=1)


    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    # print("\t+ Batch Inference Time: %s" % (inference_time))

    detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
    detections = detections[0]

    total_pred_dict = {}
    total_pred_dict['boxes'] = []
    total_pred_dict['confidence'] = []
    total_pred_dict['labels'] = []
    colors_ids = []

    pred_dict = {}
    pred_dict['boxes'] = []
    pred_dict['confidence'] = []
    pred_dict['labels'] = []
    if detections is not None:
        # Rescale boxes to original image
        detections = rescale_boxes(detections, opt.img_size, orig_img.size)

        if vis or save:
            draw = ImageDraw.Draw(orig_img)
            font = ImageFont.truetype("./calibril.ttf", 15)
            distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6']

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            # cls_conf thread
            if cls_conf < 0.8:
                continue
            if classes[int(cls_pred)] == 'car':
                continue

            x1, x2 = [overbound(i, 1280) for i in [x1, x2]]
            y1, y2 = [overbound(i, 720) for i in [y1, y2]]

            box_location = [int(i) for i in [x1, y1, x2, y2]]

            total_pred_dict['boxes'].append(box_location)
            total_pred_dict['confidence'].append(conf.item())
            total_pred_dict['labels'].append(classes[int(cls_pred)])
            colors_ids.append(int(cls_pred))
        if len(total_pred_dict['boxes']) == 0:
            return orig_img, pred_dict
        
        keep = py_cpu_softnms(np.array(total_pred_dict['boxes']), np.array(total_pred_dict['confidence']),
                              Nt=0.7)
        for i in keep:
            box_location = total_pred_dict['boxes'][i]
            conf = total_pred_dict['confidence'][i]
            label = total_pred_dict['labels'][i]

            pred_dict['boxes'].append(box_location)
            pred_dict['confidence'].append(conf)
            pred_dict['labels'].append(label)
            if vis or save:
                # Box
                draw.rectangle(xy=box_location, outline=distinct_colors[colors_ids[i]])
                draw.rectangle(xy=[l + 1. for l in box_location], outline=distinct_colors[colors_ids[i]])

                # Text
                text_size = font.getsize(label.upper())

                text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
                textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                                    box_location[1]]
                draw.rectangle(xy=textbox_location, fill=distinct_colors[colors_ids[i]])
                draw.text(xy=text_location, text=label.upper(), fill='white',
                          font=font)

                text_conf = [box_location[0] + 2., box_location[1]]
                textbox_conf = [box_location[0], box_location[1] + text_size[1], box_location[0] + text_size[0] + 4.,
                                box_location[1]]
                draw.rectangle(xy=textbox_conf, fill=distinct_colors[colors_ids[i]])
                draw.text(xy=text_conf, text='%.2f'%(conf), fill='white',
                          font=font)

    return orig_img, pred_dict


if __name__ == '__main__':
    from pprint import pprint
    from tqdm import tqdm

    # path = opt.image_folder
    path = '../data/sub_test/test/'
    save = False
    vis = True
    flip = True

    for file in tqdm(os.listdir(path)):
        if not save:
            file = '003220.jpg'

        img = Image.open(path + file)
        # img = img.transpose(Image.FLIP_LEFT_RIGHT)
        draw, pred_dict = detect(img, save, vis, flip)

        if not save:
            pprint(pred_dict)
            draw.show()

            break

        else:
            draw.save(f'output/{file.split(".")[0]}.png')
