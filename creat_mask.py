import os
import torch
import matplotlib.pyplot as plt
from xml.dom.minidom import parse
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
opj = os.path.join

parser = argparse.ArgumentParser()
parser.add_argument("--root_folder", type=str, default="../data", help="path to root  folder")
opt = parser.parse_args()


def get_wheel_mask(file):
    DOMTree = parse(file)
    collection = DOMTree.documentElement
    classes = []
    xymin = []
    xymax = []

    for object in collection.getElementsByTagName('object'):
        if object.getElementsByTagName('name')[0].childNodes[0].data == 'rear' or \
            object.getElementsByTagName('name')[0].childNodes[0].data == 'front' or \
            object.getElementsByTagName('name')[0].childNodes[0].data == 'person':

            classes.append(object.getElementsByTagName('name')[0].childNodes[0].data)

            xymin.append((int(object.getElementsByTagName('xmin')[0].childNodes[0].data),
                          int(object.getElementsByTagName('ymin')[0].childNodes[0].data)))

            xymax.append((int(object.getElementsByTagName('xmax')[0].childNodes[0].data),
                          int(object.getElementsByTagName('ymax')[0].childNodes[0].data)))

    mask = torch.zeros(720, 1280)
    for i, cls in enumerate(classes):
        if cls == 'rear':
            mask[xymin[i][1]:xymax[i][1], xymin[i][0]: xymax[i][0]] = 1
        elif cls == 'front':
            mask[xymin[i][1]:xymax[i][1], xymin[i][0]: xymax[i][0]] = 2

    return np.array(mask, dtype=int)


def mask2img(mask):
    mask = np.array(mask, dtype=int)
    img = transforms.ToPILImage()(mask)
    img = img.convert('P')
    return img



if __name__ == '__main__':
    for mode in ['train', 'test']:
        files = os.listdir(opj(opt.root_folder, mode))
        files = list(map(lambda x: x.split('.')[0], files))

        os.mkdir(opj(opt.root_folder, f'{mode}_mask'), exist_ok=True)

        for file in tqdm(files):
            mask = get_wheel_mask(opj(opt.root_folder, f'{mode}_label/{file}.xml'))
            img = mask2img(mask)
            img.save(opj(opt.root_folder, f'{mode}_mask_with_person/{file}_mask.png')
