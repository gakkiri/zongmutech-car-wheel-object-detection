import torch
from torchvision import transforms
from utils import *
# from detects.detect_yolo import detect as detect_yolo
from detects.detect_wheel import detect as detect_wheel
from detects.detect_fastrcnn import detect as detect_fsrcnn
from detects.detect_fastrcnn2 import detect as detect_fsrcnn2
from soft_nms import py_cpu_softnms as nms

import os
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
from pprint import pprint
import argparse
opj = os.path.join

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, default="../data/sub_test/test", help="path to test images folder")
parser.add_argument("--vis", type=bool, default=False, help="output visiable result")
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICE'] = '0'

num2name = {1:'car', 2:'person', 3:'truck', 4:'bus', 5:'rider', 6:'wheel', 7:'rear', 8:'front'}
name2num = {'car':1, 'person':2, 'truck':3, 'bus':4, 'rider':5, 'wheel':6, 'rear':7, 'front':8}

def is_in_keep(ids, keep):
    for i, flag in enumerate(ids):
        if flag:
            if i not in keep:
                ids[i] = False
    return ids

def car2wheel(pred_dict, keep):
    wheel_ids = (pred_dict['labels'] == 6) + (pred_dict['labels'] == 7) + (pred_dict['labels'] == 8)
    wheel_ids = is_in_keep(wheel_ids, keep)

    car_ids = pred_dict['labels'] == 1
    car_ids = is_in_keep(car_ids, keep)
    
    wheel_boxes = pred_dict['boxes'][wheel_ids]
    car_boxes = pred_dict['boxes'][car_ids]
    wheel_centers = np.array([wheel_boxes[:, 0]+(wheel_boxes[:, 2]-wheel_boxes[:, 0])/2,
                                                    wheel_boxes[:, 1]+(wheel_boxes[:, 3]-wheel_boxes[:, 1])/2]).T
    c2w = {}
    for i, box in enumerate(car_boxes):
        c2w[i] = []
        for j in range(len(wheel_centers)):
            if box[0] <= wheel_centers[j][0] and box[1] <= wheel_centers[j][1] \
                and box[2] >= wheel_centers[j][0] and box[3] >= wheel_centers[j][1]:
                c2w[i].append(wheel_boxes[j])
    return c2w

def ens(img_path):
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    # original_image = transforms.RandomHorizontalFlip(1)(original_image)

    # _, pred_dict_yolo = detect_yolo(original_image.copy())
    _, pred_dict_wheel, mask = detect_wheel(original_image.copy())
    _, pred_dict_fsrcnn = detect_fsrcnn(original_image.copy(), True, True)  # 6 -> wheel
    _, pred_dict_fsrcnn2 = detect_fsrcnn2(original_image.copy(), True, True)

    total_labels = \
                             pred_dict_wheel['labels'] + \
                             pred_dict_fsrcnn['labels'] + \
                             pred_dict_fsrcnn2['labels']
    total_boxes = \
                              pred_dict_wheel['boxes'] + \
                              pred_dict_fsrcnn['boxes'].astype('int').tolist() + \
                              pred_dict_fsrcnn2['boxes'].astype('int').tolist()
    total_conf = \
                           pred_dict_wheel['confidence'] + \
                           pred_dict_fsrcnn['confidence'].tolist() + \
                           pred_dict_fsrcnn2['confidence'].tolist()

    total_pred_dict = {}
    pred_dict = {}
    pred_dict['boxes'] = []
    pred_dict['confidence'] = []
    pred_dict['labels'] = []

    total_pred_dict['labels'] = np.array([name2num[name] for name in total_labels])
    total_pred_dict['boxes'] = np.array(total_boxes)
    total_pred_dict['confidence'] = np.array(total_conf)

    wheel_ids = \
            (total_pred_dict['labels'] == 6) + \
            (total_pred_dict['labels'] == 7) + \
            (total_pred_dict['labels'] == 8)
    wheel_labels = total_pred_dict['labels'][wheel_ids]
    wheel_boxes = total_pred_dict['boxes'][wheel_ids]
    wheel_conf = total_pred_dict['confidence'][wheel_ids]
    wheel_conf[wheel_conf == 1.] = 0.5

    for i, box in enumerate(wheel_boxes):
        if wheel_labels[i] == 6:
            cls = mask[box[1]:box[3], box[0]:box[2]].cpu()

            if cls.sum() == 0:
                continue
            elif (cls == 1).sum() > (cls == 2).sum():
                wheel_labels[i] = 7
            elif (cls == 1).sum() < (cls == 2).sum():
                wheel_labels[i] = 8

    wheel_keep = nms(wheel_boxes.copy(), wheel_conf.copy(), Nt=0.01, thresh=0.01)
    wheel_labels = wheel_labels[wheel_keep]
    wheel_boxes = wheel_boxes[wheel_keep]
    wheel_conf = wheel_conf[wheel_keep]

    total_pred_dict['labels'] = total_pred_dict['labels'][~wheel_ids]
    total_pred_dict['boxes'] = total_pred_dict['boxes'][~wheel_ids]
    total_pred_dict['confidence'] = total_pred_dict['confidence'][~wheel_ids]

    total_pred_dict['labels'] = np.concatenate([total_pred_dict['labels'], wheel_labels])
    total_pred_dict['boxes'] = np.concatenate([total_pred_dict['boxes'], wheel_boxes])
    total_pred_dict['confidence'] = np.concatenate([total_pred_dict['confidence'], wheel_conf])
    

    # total_pred_dict['confidence'][total_pred_dict['labels'] == 6] = 0.5

    if len(total_boxes) == 0:
        return original_image, pred_dict

    if len(total_boxes) <= 15:
        keep = nms(total_pred_dict['boxes'].copy(), total_pred_dict['confidence'].copy(), Nt=0.3, thresh=0.01, method=3)
    else:
        keep = nms(total_pred_dict['boxes'].copy(), total_pred_dict['confidence'].copy(), Nt=0.5, thresh=0.75, method=2)

    for i, label in enumerate(total_pred_dict['labels']):
        if label in [7, 8] and i not in keep:
            keep = np.concatenate([keep, np.array([i])])
            
    c2w = car2wheel(total_pred_dict, keep)

    for wheel_boxes in c2w.values():
        if len(wheel_boxes) == 2:
            label1_ids = np.sum(total_pred_dict['boxes'] == wheel_boxes[0], 1) == 4
            label2_ids = np.sum(total_pred_dict['boxes'] == wheel_boxes[1], 1) == 4

            label1_ids = is_in_keep(label1_ids, keep)
            label2_ids = is_in_keep(label2_ids, keep)
            labels = total_pred_dict['labels'][label1_ids +  label2_ids]
            if 6 in labels and 7 in labels:
                if total_pred_dict['labels'][label1_ids] == 6:
                    total_pred_dict['labels'][label1_ids] = 8
                elif total_pred_dict['labels'][label2_ids] == 6:
                    total_pred_dict['labels'][label2_ids] = 8
                    
            elif 6 in labels and 8 in labels:
                if total_pred_dict['labels'][label1_ids] == 6:
                    total_pred_dict['labels'][label1_ids] = 7
                elif total_pred_dict['labels'][label2_ids] == 6:
                    total_pred_dict['labels'][label2_ids] = 7
                    
    total_pred_dict['labels'] = [num2name[i] for i in total_pred_dict['labels']]
    del_wheel_ids = []
    for i, label in enumerate(total_pred_dict['labels']):
        if label == 'wheel':
            del_wheel_ids.append(i)

    final_annotated_image = original_image
    draw = ImageDraw.Draw(final_annotated_image)
    # font = ImageFont.truetype("./calibril.ttf", 15)

    for i in keep:
        if i in del_wheel_ids:
            continue
        # Boxes
        box_location = [round(j) for j in total_pred_dict['boxes'][i]]
        draw.rectangle(xy=box_location, outline=label_color_map[total_pred_dict['labels'][i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            total_pred_dict['labels'][i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        # text_size = font.getsize(total_labels[i].upper())
        # text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        # textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
        #                     box_location[1]]
        # draw.rectangle(xy=textbox_location, fill=label_color_map[total_labels[i]])
        # draw.text(xy=text_location, text=total_labels[i].upper(), fill='white',
        #           font=font)

        pred_dict['boxes'].append(box_location)
        pred_dict['confidence'].append(total_pred_dict['confidence'][i])
        pred_dict['labels'].append(total_pred_dict['labels'][i])

    return final_annotated_image, pred_dict


if __name__ == '__main__':
    from pprint import pprint
    DEBUG = False
    OUTPUT = opt.vis

    if DEBUG:
        # img_path = opj(opt.image_folder, '003276.jpg')  # 000374
        img_path = '../1.jpg'
        _, total_pred_dict = ens(img_path)
        _.show()
        pprint(total_pred_dict)

    else:
        # to submission txt
        test_path = opt.image_folder
        test_files = os.listdir(test_path)
        w = ''
        for file in tqdm(test_files):
            try:
                img_path = opj(test_path, file)

                _, total_pred_dict = ens(img_path)
                if OUTPUT:
                    _.save(f'output/{file.split(".")[0]}.png')

                pred_dict = to_submission(total_pred_dict)

                for i, cls in enumerate(pred_dict['labels']):
                    anno = pred_dict['boxes'][i]
                    anno = ' '.join([str(b) for b in anno])
                    w += f'{file.split(".")[0]} {cls} {anno}\n'

            except Exception as e:
                print('Error file:', file)
                print(e)
                exit()

        with open('submit.txt', 'w') as f:
            f.write(w)




