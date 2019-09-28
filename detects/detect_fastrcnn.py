import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
from pprint import pprint

from utils import *
from soft_nms import py_cpu_softnms as nms

def _get_instance_segmentation_model(num_classes, pretrained=False):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)   
    return model

print(f'Load FaterRCNN-ResNet50FPN...')
checkpoint = 'checkpoints/fasterrcnn_gcp_120.pth.tar'
model = _get_instance_segmentation_model(num_classes=len(label_map), pretrained=False).cuda()
model.load_state_dict(torch.load(checkpoint)['model'])
model.eval()
print(f'FaterRCNN-ResNet50FPN Done\n')


def detect(img, use_wheels=True, flip=True):
    draw_img = img.copy()
    if flip:
        flip_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
    img = transforms.Resize((800, 1280))(img)
    img = transforms.ToTensor()(img)
    # img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
    img = img.unsqueeze(0)
    img = img.cuda()
    # detections = model(img)[0]

    if flip:
        flip_img = transforms.Resize((800, 1280))(flip_img)
        flip_img = transforms.ToTensor()(flip_img)
        # img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        flip_img = flip_img.unsqueeze(0)
        flip_img = flip_img.cuda()

        inputs = torch.cat([img, flip_img], dim=0)

        detections = model(inputs)
        flip_detections = detections[1]
        detections = detections[0]
        
        flip_boxes = flip_detections['boxes']
        flip_boxes[:, 0] = 1280 - flip_boxes[:, 0] - 1
        flip_boxes[:, 2] = 1280 - flip_boxes[:, 2] - 1
        flip_boxes = flip_boxes[:, [2, 1, 0, 3]]

        detections['boxes'] = torch.cat([detections['boxes'], flip_boxes], dim=0)
        detections['scores'] = torch.cat([detections['scores'], flip_detections['scores']], dim=0)
        detections['labels'] = torch.cat([detections['labels'], flip_detections['labels']], dim=0)
    else:
        detections = model(img)[0]
        
    keep = nms(detections['boxes'].cpu().detach().numpy(),
                           detections['scores'].cpu().detach().numpy(),
                           Nt=0.5,
                           thresh=0.55,
                           method=2)
    if not use_wheels:
        for i, keep_i in enumerate(keep):
            if detections['labels'].cpu().detach().numpy().tolist()[keep_i] == 6:
                keep[i] = 100
        keep = keep[keep != 100]  
    
    pred_dict = {}
    pred_dict['boxes'] = detections['boxes'].cpu().detach().numpy()[keep]
    pred_dict['labels'] = [rev_label_map[i] for i in detections['labels'].cpu().detach().numpy()[keep]]
    pred_dict['confidence'] = detections['scores'].cpu().detach().numpy()[keep]

    pred_dict['boxes'][:, 1] = (pred_dict['boxes'][:, 1] / 800) * 720
    pred_dict['boxes'][:, 3] = (pred_dict['boxes'][:, 3] / 800) * 720

    draw = ImageDraw.Draw(draw_img)
    for i in pred_dict['boxes']:
        box = i.tolist()
        box = [round(j) for j in box]
        draw.rectangle(box, outline='red')

    del draw
    return draw_img, pred_dict

if __name__ == '__main__':
    from pprint import pprint
    
    file = '../data/sub_test/test/002966.jpg'
    img = Image.open(file)
    
    draw, pred_dict = detect(img)
    
    pprint(pred_dict)
    draw.show()
