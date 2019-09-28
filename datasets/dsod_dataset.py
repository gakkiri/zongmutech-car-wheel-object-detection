import os, random

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as FT
from PIL import Image
import torchvision.transforms as transforms

from utils import *
opj = os.path.join

# v2classes = {
#     0: 'background',
#     1: 'car',
#     2: 'person',
#     3: 'truck',
#     4: 'bus',
#     5: 'rider',
#     6: 'rear',
#     7: 'front',
# }
#
# classes2v = dict([(v,k) for (k,v) in v2classes.items()])

class HorizontalFlip(object):
    def __call__(self, img, mask):
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


class mydateset(Dataset):  # MARK
    def __init__(self, root='data', mode='train', transform=False):
        assert mode in ['train', 'test', 'oversample'], f'mode only choose "train", "test" and "oversample"'

        self.root = root
        self.transform = transform
        self.mode = mode
        self.path = f'{self.mode}'

        if self.mode in ['train', 'test']:
            self.img_files = os.listdir(opj(self.root, self.path))

    def __getitem__(self, idx):
        img = self.img_files[idx]
        img_name = img.split('.')[0]
        img = Image.open(opj(self.root, self.path, img))
        img = img.convert('RGB')

        mask = Image.open(opj(self.root, self.mode +'_mask', img_name + '_mask.png'))

        if self.mode == 'train' or self.mode == 'test':
            if self.mode == 'train':
                # print(img_name)
                classes, xymin, xymax = get_inf(opj(self.root, self.path + '_label', img_name + '.xml'))
            else:
                classes, xymin, xymax, difficulties = get_inf(opj(self.root, self.path + '_label', img_name + '.xml'),
                                                              get_diff=True)
            label = [label_map[i] for i in classes]
            label = torch.FloatTensor(label)
            boxes = [list(xymin[i] + xymax[i]) for i in range(len(label))]
            boxes = torch.FloatTensor(boxes)

            if self.transform and self.mode == 'train':
                # filp
                if random.random() < 0.5:
                    img, mask = HorizontalFlip()(img, mask)
                    filp_boxes = boxes
                    filp_boxes[:, 0] = img.width - boxes[:, 0] - 1
                    filp_boxes[:, 2] = img.width - boxes[:, 2] - 1
                    filp_boxes = filp_boxes[:, [2, 1, 0, 3]]
                    boxes = filp_boxes

                # photometric distort
                img = self.photometric_distort(img)

                # random crop
                img, boxes, label, mask = self.random_crop(img, boxes, label, mask)
                img = transforms.ToPILImage()(img)
                mask = transforms.ToPILImage()(mask)

                # resize
                new_boxes = self.box_resize(boxes, img)
                img = transforms.Resize((576, 1024))(img)
                mask = transforms.Resize((576, 1024))(mask)

                # totensor && norm
                img = transforms.ToTensor()(img)
                img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

                return img, new_boxes, label, torch.from_numpy(np.array(mask)).unsqueeze(0)

            elif self.mode == 'test':
                # resize
                new_boxes = self.box_resize(boxes, img)
                img = transforms.Resize((576, 1024))(img)
                mask = transforms.Resize((576, 1024))(mask)

                # totensor && norm
                img = transforms.ToTensor()(img)
                img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

                return img, new_boxes, label, torch.from_numpy(np.array(mask)).unsqueeze(0), difficulties


    def __len__(self):
        return len(self.img_files)

    def random_crop(self, image, boxes, labels, mask):
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        original_h = image.size(1)
        original_w = image.size(2)
        # Keep choosing a minimum overlap until a successful crop is made
        while True:
            # Randomly draw the value for minimum overlap
            min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping

            # If not cropping
            if min_overlap is None:
                return image, boxes, labels, mask

            # Try up to 50 times for this choice of minimum overlap
            # This isn't mentioned in the paper, of course, but 50 is chosen in paper authors' original Caffe repo
            max_trials = 50
            for _ in range(max_trials):
                # Crop dimensions must be in [0.3, 1] of original dimensions
                # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in the authors' repo
                min_scale = 0.3
                scale_h = random.uniform(min_scale, 1)
                scale_w = random.uniform(min_scale, 1)
                new_h = int(scale_h * original_h)
                new_w = int(scale_w * original_w)

                # Aspect ratio has to be in [0.5, 2]
                aspect_ratio = new_h / new_w
                if not 0.5 < aspect_ratio < 2:
                    continue

                # Crop coordinates (origin at top-left of image)
                left = random.randint(0, original_w - new_w)
                right = left + new_w
                top = random.randint(0, original_h - new_h)
                bottom = top + new_h
                crop = torch.FloatTensor([left, top, right, bottom])  # (4)

                # Calculate Jaccard overlap between the crop and the bounding boxes
                overlap = find_jaccard_overlap(crop.unsqueeze(0),
                                               boxes)  # (1, n_objects), n_objects is the no. of objects in this image
                overlap = overlap.squeeze(0)  # (n_objects)

                # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
                if overlap.max().item() < min_overlap:
                    continue

                # Crop image and mask
                new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)
                new_mask = mask[:, top:bottom, left:right]  # (1, new_h, new_w)

                # Find centers of original bounding boxes
                bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

                # Find bounding boxes whose centers are in the crop
                centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                        bb_centers[:,
                        1] < bottom)  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

                # If not a single bounding box has its center in the crop, try again
                if not centers_in_crop.any():
                    continue

                # Discard bounding boxes that don't meet this criterion
                new_boxes = boxes[centers_in_crop, :]
                new_labels = labels[centers_in_crop]

                # Calculate bounding boxes' new coordinates in the crop
                new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
                new_boxes[:, :2] -= crop[:2]
                new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
                new_boxes[:, 2:] -= crop[:2]

                return new_image, new_boxes, new_labels, new_mask

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

    def box_resize(self, boxes, img, dims=(576, 1024), return_percent_coords=True):
        new_boxes = []
        old_dims = torch.FloatTensor([img.width, img.height, img.width, img.height]).unsqueeze(0)
        for box in boxes:
            new_box = box / old_dims
            if not return_percent_coords:
                new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
                new_box = new_box * new_dims
            new_boxes.append(new_box)
        return new_boxes

    def collate_fn(self, batch):
        if self.mode == 'train':
            images = list()
            boxes = list()
            labels = list()
            masks = list()

            for b in batch:
                images.append(b[0])
                boxes.append(b[1])
                labels.append(b[2])
                masks.append(b[3])

            images = torch.stack(images, dim=0)

            return images, boxes, labels, masks

        elif self.mode == 'test':
            images = list()
            boxes = list()
            labels = list()
            masks = list()
            difficulties = list()

            for b in batch:
                images.append(b[0])
                boxes.append(b[1])
                labels.append(b[2])
                masks.append(b[3])
                difficulties.append(b[4])

            images = torch.stack(images, dim=0)

            return images, boxes, labels, masks, difficulties


def dataloader_test():
    ds = mydateset('data', mode='train', transform=True)
    dataloader = DataLoader(
            ds, batch_size=2, shuffle=True, num_workers=1, collate_fn=ds.collate_fn
        )

    a = next(iter(dataloader))
    print(a[0].shape)
    print(a[1])
    print(a[2])
    print(a[3][0].shape)

    boxes = [torch.cat(b).to(device) for b in a[1]]
    labels = [l.to(device) for l in a[2]]
    # masks = torch.cat(a[3], dim=0)
    masks = torch.cat([m.unsqueeze(0) for m in a[3]])
    print(masks.shape)
    print(np.unique(masks.detach().numpy()))


if __name__ == '__main__':
    dataloader_test()
