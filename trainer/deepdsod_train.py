import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from nets.deepdsod_model import DSOD
from loss import MultiBoxLoss
from datasets.dsod_dataset import mydateset
from os.path import exists
from utils import *
# {'car': 1, 'person': 2, 'truck': 3, 'bus': 4, 'rider': 5, 'rear': 6, 'front': 7}

# Data parameters
keep_difficult = True  # use objects considered difficult to detect?
use_focalloss = False

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(opt):
    """
    Training and validation.
    """
    global epochs_since_improvement, start_epoch, label_map, best_loss, epoch, checkpoint, lr_scheduler
    epochs_since_improvement = opt['epochs_since_improvement']
    start_epoch = opt['start_epoch']
    best_loss = opt['best_loss']
    checkpoint = opt['checkpoint']
    lr_scheduler = opt['lr_scheduler']

    batch_size = opt['batch_size']
    epochs = opt['epochs']
    lr = opt['lr']
    momentum = opt['momentum']
    weight_decay = opt['weight_decay']
    grad_clip = opt['grad_clip']
    workers = opt['workers']
    print_freq = opt['print_freq']

    root = opt['root']

    # Initialize model or load checkpoint
    if checkpoint is None:
        model = DSOD(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_loss = checkpoint['best_loss']
        print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
        model = checkpoint['model']

        # optimizer = checkpoint['optimizer']
        # or 
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        optimizer = torch.optim.SGD(model.parameters(),
                                                              lr=lr, momentum=momentum, weight_decay=weight_decay)

        print('Learning Rate: ', optimizer.param_groups[-1]['lr'])

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=20, verbose=True
    )
    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy, use_focalloss=use_focalloss).to(device)

    # Custom dataloaders
    train_dataset = mydateset(root='../data', transform=True)
    val_dataset = mydateset(root='../data', mode='test')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                             collate_fn=val_dataset.collate_fn, num_workers=workers,
                                             pin_memory=True)
    # Epochs
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # One epoch's validation
        val_loss = validate(val_loader=val_loader,
                            model=model,
                            criterion=criterion)

        # Did validation loss improve?
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        if lr_scheduler is not None:
            lr_scheduler.step(best_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))

        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, val_loss, best_loss, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels, masks) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)
        boxes = [torch.cat(b).to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        masks = torch.cat([m.unsqueeze(0) for m in masks]).to(device)

        # Forward prop.
        predicted_locs, predicted_scores, segm_score = model(images)
        # print(predicted_locs.shape, predicted_scores.shape, segm_score.shape)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, segm_score, boxes, labels, masks)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


def validate(val_loader, model, criterion):
    """
    One epoch's validation.

    :param val_loader: DataLoader for validation data
    :param model: model
    :param criterion: MultiBox loss
    :return: average validation loss
    """
    model.eval()  # eval mode disables dropout

    batch_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    # Prohibit gradient computation explicity because I had some problems with memory
    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, masks, difficulties) in enumerate(val_loader):

            # Move to default device
            images = images.to(device)
            boxes = [torch.cat(b).to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            masks = torch.cat([m.unsqueeze(0) for m in masks]).to(device)

            # Forward prop.
            predicted_locs, predicted_scores, segm_score = model(images)

            # Loss
            loss = criterion(predicted_locs, predicted_scores, segm_score, boxes, labels, masks)

            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % print_freq == 0:
                print('[{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses))

    print('\n * LOSS - {loss.avg:.3f}\n'.format(loss=losses))

    return losses.avg


if __name__ == '__main__':
    train()
