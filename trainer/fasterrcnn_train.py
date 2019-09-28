from utils import *
from datasets.fasterrcnn_dataset import mydateset

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from PIL import Image

import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import random


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def _get_instance_segmentation_model(num_classes, pretrained=False):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, epoch, print_freq):
    all_loss = []
    all_cls_loss = []
    all_reg_loss = []

    model.train()
    for step, (images, boxes, labels) in enumerate(data_loader):
        targets = []
        for i in range(images.shape[0]):
            t = {}
            t['boxes'] = torch.cat([j.cuda() for j in boxes[i]])
            t['labels'] = torch.cat([j.unsqueeze(0).cuda() for j in labels[i]])
            targets.append(t)
        images = images.cuda()

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        all_loss.append(losses.item())
        all_cls_loss.append(loss_dict['loss_classifier'].item())
        all_reg_loss.append(loss_dict['loss_box_reg'].item())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if step % print_freq == 0:
            _lr = optimizer.param_groups[0]["lr"]
            print(f'Epoch: [{epoch}][{step}/{len(data_loader)}]  \tTotal_loss: {str(np.average(all_loss))[:6]}',
                  f'\tlr: {_lr}',
                  f'\tLoss classifier: {str(np.average(all_cls_loss))[:6]}',
                  f'\tLoss regression: {str(np.average(all_reg_loss))[:6]}')

        return np.average(all_loss)


def evaluate(model, data_loader, print_freq=200):
    all_loss = []
    all_cls_loss = []
    all_reg_loss = []

    with torch.no_grad():
        for step, (images, boxes, labels, _) in enumerate(data_loader):
            targets = []
            for i in range(images.shape[0]):
                t = {}
                t['boxes'] = torch.cat([j.cuda() for j in boxes[i]])
                t['labels'] = torch.cat([j.unsqueeze(0).cuda() for j in labels[i]])
                targets.append(t)
            images = images.cuda()

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            all_loss.append(losses.item())
            all_cls_loss.append(loss_dict['loss_classifier'].item())
            all_reg_loss.append(loss_dict['loss_box_reg'].item())

            if step % print_freq == 0:
                print(f'[{step}/{len(data_loader)}]  \tTotal_loss: {str(np.average(all_loss))[:6]}',
                      f'\tLoss classifier: {str(np.average(all_cls_loss))[:6]}',
                      f'\tLoss regression: {str(np.average(all_reg_loss))[:6]}')

    return np.average(all_loss)


def train(epochs, print_freq):
    lr_scheduler = None
    since_best_loss = 0
    best_loss = 1e2
    bs = 2

    model = _get_instance_segmentation_model(num_classes=len(label_map), pretrained=False)
    model = model.cuda()

    ds = mydateset('../data', mode='train', transform=True)
    dataloader = DataLoader(
            ds, batch_size=bs, shuffle=True, num_workers=4, collate_fn=ds.collate_fn
        )

    eval_ds = mydateset('../data', mode='test')
    eval_dataloader = DataLoader(
            eval_ds, batch_size=bs, shuffle=False, num_workers=4, collate_fn=eval_ds.collate_fn
        )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=5e-4)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.9, patience=3, verbose=True
    )

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, optimizer, dataloader, epoch, print_freq=print_freq)
        eval_loss = evaluate(model, eval_dataloader)

        lr_scheduler.step()
        print('* Train LOSS: ', str(train_loss)[:6], '\tEval LOSS: ', str(eval_loss)[:6])

        if lr_scheduler is not None:
            lr_scheduler.step(eval_loss)
        
        if best_loss > eval_loss:
            since_best_loss = 0
            best_loss = eval_loss
            is_best = 1
        else:
            since_best_loss += 1
            print('Since best loss epoch: ', since_best_loss)
            is_best = 0

        save_checkpoint(epoch, since_best_loss, model, optimizer, eval_loss, best_loss, is_best)


if __name__ == '__main__':
    train(epochs=200, print_freq=200)

