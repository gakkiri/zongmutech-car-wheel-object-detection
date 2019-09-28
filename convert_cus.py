import os
import shutil
from xml.etree import ElementTree
import numpy as np
from tqdm import tqdm
opj = os.path.join

voc_labels = ('car', 'person', 'truck', 'bus', 'rider', 'wheel')
label_map = {k: v for v, k in enumerate(voc_labels)}


def parse_annotation(annotation_path):
    tree = ElementTree.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()
    for object in root.iter('object'):

        difficult = int(object.find('difficult').text == '1')

        label = object.find('name').text.lower().strip()
        if label == 'rear' or label == 'front':
            label = 'wheel'
        if label not in label_map:
            continue

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)


    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}

root = '../data'
for mode in ['train', 'test']:
    W = 1280
    H = 720

    paths = []
    for file in tqdm(os.listdir(opj(root, mode))):
        file = file.split('.')[0]

        path = opj(root, mode, f'{file}.jpg')
        # path = f'data/custom/images/{file}.jpg'
        
        # shutil.copyfile(opj(root, mode, file+'.jpg'), f'images/{file}.jpg')
        
        paths.append(path)
        label_path = opj(root, mode+'_label', file+'.xml')
        label = parse_annotation(label_path)
        if len(label['boxes']) == 0:
            continue

        boxes = np.array(label['boxes'])
        # (x1, y1, x2, y2) to (cx, cy, w, h)
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        cx = boxes[:, 0] + (w / 2)
        cy = boxes[:, 1] + (h / 2)
        boxes = np.concatenate([[cx], [cy], [w], [h]]).T

        boxes[:, 0] /= W
        boxes[:, 1] /= H
        boxes[:, 2] /= W
        boxes[:, 3] /= H
        
        label = label['labels']
        
        with open(f'yolov3/labels/{file}.txt', 'w') as f:
            s = ''
            for i in range(len(label)):
                anno = ' '.join([str(i) for i in boxes[i].tolist()])
                s += f'{label[i]} {anno}\n'
                
            f.write(s)

    with open(f'yolov3/{mode}.txt', 'w') as f:
        s = ''
        for path in paths:
            s += path + '\n'
        
        f.write(s)
