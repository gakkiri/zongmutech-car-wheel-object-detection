from loss import cross_entropy2d
from datasets.deeplab_dataset import myds
from nets.seg_model import DeepLabv3_plus

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch import nn

from utils import *

##import sys
##sys.path.append('./')
##from sycnbn import nn as NN
BatchNorm2d = nn.BatchNorm2d


def train(opt):
    
    ds = myds(root=opt.root)
    dl = DataLoader(ds, batch_size=opt['batch_size'], shuffle=True)

    valid_ds = myds(root=opt.root, mode='test')
    valid_dl = DataLoader(valid_ds, batch_size=2, shuffle=False)


    model = DeepLabv3_plus(nInputChannels=3, n_classes=3, os=16, pretrained=False, _print=True)
    # model = torch.load('best_deeplabv3.pth')
    model = model.cuda()

##    if opt['set_BN_momentum']:
##        for i, (name, layer) in enumerate(model.named_modules()):
##            if isinstance(layer, BatchNorm2d):
##                layer.momentum = 0.1*opt['batch_size']/64
    
##    model = torch.nn.DataParallel(model)
    model.train()

    best_loss = 1e2
    criterion = cross_entropy2d
    last_best = 0

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=opt['lr'],
                                momentum=opt['momentum'],
                                weight_decay=opt['weight_decay'])

    for epoch in range(opt['epochs']):
        model.train()
        
        train_loss = []
        for step, (imgs, labels) in enumerate(dl):
            imgs = imgs.cuda()
            labels = labels.cuda()

            outputs = model(imgs)

            loss = criterion(outputs, labels, ignore_index=255, size_average=True, batch_average=True)
            # loss = criterion(outputs, labels)
            train_loss.append(loss.item())

            loss.backward()
            if (step+1) % opt['AccumulateStep'] == 0:
                optimizer.step()
                optimizer.zero_grad()

            if step % 100 == 0:
                print(f'epoch: {epoch}/{opt["epochs"]}', f'  \tstep: {step}/{len(dl)}', f'   \tloss: {loss.item()}')

        print(f'epoch[{epoch}/{opt["epochs"]}[', f'  \tstep[{step}/{len(dl)}]', f'   \tloss: {str(np.average(train_loss).item())[:6]}')

        model.eval()
        valid_loss = []
        for step, (imgs, labels) in enumerate(valid_dl):
            imgs = imgs.cuda()
            labels = labels.cuda()
        
            outputs = model(imgs)
            loss = criterion(outputs, labels, ignore_index=255, size_average=True, batch_average=True)
            valid_loss.append(loss.item())
        
            if step % 100 == 0:
                print(f'step: {step}/{len(valid_dl)}', f'    \tvalid loss: {loss.item()}')
        valid_loss = np.average(valid_loss)
        train_loss = np.average(train_loss)
        print(f'train loss: {train_loss}')
        
        if valid_loss < best_loss:
            last_best = 0
            best_loss = valid_loss
            is_best = 1
            print('and best loss!')
        else:
            last_best += 1
            is_best = 0
            print(f'last best {last_best}\n')
        save_checkpoint(epoch, last_best, model, optimizer, best_loss, is_best)


if __name__ == '__main__':
    epochs = 100
    hyperparameters = {
        'lr': 1e-2,
        'epochs': epochs,
        'decay_step': epochs*0.1,
        'momentum': 0.9,
        'weight_decay': 1e-5,
        'batch_size': 2,
        'set_BN_momentum': True,
        'AccumulateStep': 64//8,
    }
    train(hyperparameters)

