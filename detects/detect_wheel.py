import torch
from PIL import Image, ImageDraw
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

import numpy as np

from utils import *
from nets.seg_model import DeepLabv3_plus


model = DeepLabv3_plus(nInputChannels=3, n_classes=3, os=16,
                                               pretrained=False, _print=False, fpn=False)
print(f'Load DeepLabv3...')
checkpoint = torch.load('checkpoints/resnet101_seg.pth.tar')
state_dict = checkpoint['model']
model = model.cuda()
model.load_state_dict(state_dict)
model.eval()
print(f'DeepLabv3 Done\n')
label_color_map = {'rear': 'blue', 'front': 'red'}

def detect(img, vis=False):
      draw_img = img.copy()
      tsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
          ])

      img = tsfm(img).unsqueeze(0).cuda()
      pred = model(img)

      val, decode_mask = torch.max(pred, 1)
      index = decode_mask.cpu().detach().numpy().squeeze().astype('int32')

      _map = decode_segmap(index)
      if vis:
            plt.imshow(_map)
            plt.show()

      imgray = cv2.cvtColor(np.asarray(_map, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
      ret, thresh=cv2.threshold(imgray, 1, 255, cv2.THRESH_BINARY_INV)
      countours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

      pred_dict = {}
      pred_dict['labels'] = []
      pred_dict['boxes'] = []
      pred_dict['confidence'] = []

      n_wheels = len(countours)-1
      if n_wheels == 0:
          return draw_img, pred_dict, decode_mask.squeeze()
      # print('wheel numbers: ', n_wheels)
      xymin = []
      xymax = []
      for i in range(1, len(countours)):
            x, y, w, h = cv2.boundingRect(countours[i])
            xymin.append((x, y))
            xymax.append((x+w, y+h))

      xymin = np.array(xymin)
      xymax = np.array(xymax)
      boxes = np.concatenate([xymin, xymax], 1)

      clses = []
      new_boxes = []
      for i in range(n_wheels):
            box = list(map(lambda x:int(x), boxes[i]))
            cls = decode_mask.squeeze()[box[1]:box[3], box[0]:box[2]].cpu()

            if (cls==1).sum() + (cls==2).sum() <= 20:
                  continue
            elif (cls == 1).sum() >= (cls == 2).sum():
                  clses.append('rear')
            elif (cls == 1).sum() < (cls == 2).sum():
                  clses.append('front')
            new_boxes.append(box)

      draw = ImageDraw.Draw(draw_img)
      for i in range(len(clses)):
            location = new_boxes[i]
            draw.rectangle(xy=location, outline=label_color_map[clses[i]])
            draw.rectangle(xy=[l + 1. for l in location], outline=label_color_map[clses[i]])

            pred_dict['labels'].append(clses[i])
            pred_dict['boxes'].append(location)
            pred_dict['confidence'].append(1)

      del draw
      return draw_img, pred_dict, decode_mask.squeeze()

if __name__ == '__main__':
      from pprint import pprint
      
      img_path = '../data/sub_test/test/000000.jpg'
      img = Image.open(img_path, mode='r').convert('RGB')
      
      draw, pred_dict, _ = detect(img)

      draw.show()
      pprint(pred_dict)
