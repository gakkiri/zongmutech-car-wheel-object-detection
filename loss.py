import torch
from torch import nn
import numpy as np
from utils import *
import torch.nn.functional as F

def cross_entropy2d(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
    n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)# (batchsize, 1, 512, 512) -> (batchsize, 512, 512)
    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, size_average=False)
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(), ignore_index=ignore_index, size_average=False)
    loss = criterion(logit, target.long())

    if size_average:
        loss /= (h * w)

    if batch_average:
        loss /= n

    return loss


class FocalLoss(nn.Module):
    def __init__(self, n_cls=21, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.n_cls = n_cls

    def forward(self, logit, target):
        logit = logit.cpu()
        target = target.cpu()
        target = torch.eye(self.n_cls)[target.data.cpu()]  # to one hot
        target = target.float()

        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        loss = loss.cuda()
        if len(loss.size()) == 2:
            loss = loss.sum(dim=1)
        return loss.mean()

class MultiBoxLoss(nn.Module):
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1., beta=1., use_focalloss=False):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.beta = beta
        self.use_focalloss = use_focalloss

        self.smooth_l1 = nn.L1Loss()

        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
        self.FL = FocalLoss()

        self.ce2d = cross_entropy2d

    def forward(self, predicted_locs, predicted_scores, predicted_masks, boxes, labels, masks):

        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, n_priors, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, n_priors)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)
            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, n_priors)

            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (n_priors)

            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)
            overlap_for_each_prior[prior_for_each_object] = 1.

            label_for_each_prior = labels[i][object_for_each_prior]  # (n_priors)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (n_priors)

            true_classes[i] = label_for_each_prior
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (n_priors, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732)

        # LOCALIZATION LOSS
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & n_priors)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS
        if not self.use_focalloss:
            # ------------- OHEM ----------------
            n_positives = positive_priors.sum(dim=1)  # (N)
            n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

            conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * n_priors)
            conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

            conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

            conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
            conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
            conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
            hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, n_priors)
            hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
            conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

            conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar
        else:
            # ---------- Focal Loss ----------------
            conf_loss = self.FL(predicted_scores.view(-1, n_classes), true_classes.view(-1))

        # SEGMANTATION LOSS
        mask_loss = self.ce2d(predicted_masks, masks)

        # TOTAL LOSS
        # print(conf_loss, loc_loss, mask_loss)

        return conf_loss + self.alpha * loc_loss + self.beta * mask_loss
