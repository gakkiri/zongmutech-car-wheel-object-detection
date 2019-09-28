import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as FT

import os
from random import randint
import numpy as np
import random

from utils import decode_segmap
opj = os.path.join


class RandomHorizontalFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return img, mask


class RandomRotate(object):
    def __init__(self, degree=15):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return img, mask


class myds(Dataset):
    def __init__(self, root='data', mode='train'):
        assert mode == 'train' or mode == 'test', 'mode can choose train or test only!'

        self.root = root
        self.mode = mode
        self.img_files = os.listdir(opj(self.root, self.mode))
        self.tsfm = transforms.Compose([
            transforms.Resize((720, 1280)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx].split('.')[0]
        img = Image.open(opj(self.root, f'{self.mode}/{img_name}.jpg'))
        mask = Image.open(opj(self.root, f'{self.mode}_mask/{img_name}_mask.png'))

        if self.mode == 'train':
            img, mask = RandomRotate()(img, mask)
            img, mask = RandomHorizontalFlip()(img, mask)
            img = self.photometric_distort(img)

        img = self.tsfm(img)

        mask = transforms.Resize((720, 1280))(mask)
        mask = torch.from_numpy(np.array(mask))

        return img, mask

    def photometric_distort(self, image):
        """
        Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.

        :param image: image, a PIL Image
        :return: distorted image
        """
        new_image = image

        distortions = [FT.adjust_brightness,
                       FT.adjust_contrast,
                       FT.adjust_saturation,
                       FT.adjust_hue]

        random.shuffle(distortions)

        for d in distortions:
            if random.random() < 0.5:
                if d.__name__ is 'adjust_hue':
                    # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                    adjust_factor = random.uniform(-18 / 255., 18 / 255.)
                else:
                    # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                    adjust_factor = random.uniform(0.5, 1.5)

                # Apply this distortion
                new_image = d(new_image, adjust_factor)

        return new_image


if __name__ == '__main__':
    ds = myds()
    dl = DataLoader(ds)

    a = next(iter(dl))
    for i in a:
        print(np.unique(i.detach().numpy()))
