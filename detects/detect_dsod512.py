from torchvision import transforms
from utils import *
from soft_nms import py_cpu_softnms
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
# checkpoint = 'checkpoint.pth.tar'
checkpoint = 'checkpoints/DSOD512.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
best_loss = checkpoint['best_loss']
print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((512, 512))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect(original_image, min_score, max_overlap, top_k, suppress=None, vis=False):
    """

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores, segm_score = model(image.unsqueeze(0))


    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)

    det_boxes = torch.clamp(det_boxes, 0, 1)
    percent_boxes = det_boxes
    det_boxes = det_boxes * original_dims
    # print(percent_boxes)
    # print(det_boxes)

    without_wheel = det_labels[0] != 6
    det_labels[0] = det_labels[0][without_wheel]
    det_scores[0] = det_scores[0][without_wheel]
    det_boxes = det_boxes[without_wheel]
    percent_boxes = percent_boxes[without_wheel]
    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    # print(det_labels)
    # print(det_scores)
    # print(len(det_labels), det_scores[0].shape, det_boxes.shape)

    # decode mask
    # print(segm_score.shape)
    val, decode_mask = torch.max(segm_score, 1)

    # locate wheels
    index = decode_mask.cpu().detach().numpy().squeeze().astype('int32')
    vis_mask = decode_segmap(index)
    if vis:
        plt.imshow(vis_mask)
        plt.show()
    imgray = cv2.cvtColor(np.asarray(vis_mask, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    ret, thresh=cv2.threshold(imgray, 1, 255, cv2.THRESH_BINARY_INV)
    countours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    n_wheels = len(countours)-1
    # print('wheel numbers: ', n_wheels)
    if n_wheels > 0:
        w_xymin = []
        w_xymax = []
        for i in range(1, len(countours)):
            det_labels.append('wheel')
            x, y, w, h = cv2.boundingRect(countours[i])
            w_xymin.append((x, y))
            w_xymax.append((x+w, y+h))
            # print(x, y, x + w, y + h)
        # get percent x, y
        percent_xymin = np.array(w_xymin) / index.shape[0]
        percent_xymax = np.array(w_xymax) / index.shape[0]

        # w_xy = np.concatenate([np.array(w_xymin), np.array(w_xymax)], 1)
        w_perc_xy = np.concatenate([percent_xymin, percent_xymax], 1)

        percent_boxes = torch.cat([percent_boxes, torch.tensor(w_perc_xy).float()])

        # 512,512 to 720p
        w, h = original_image.size
        wheels_boxes = []
        for p_box in w_perc_xy:
            wheels_boxes.append([p_box[0]*w, p_box[1]*h, p_box[2]*w, p_box[3]*h])
        det_boxes = torch.cat([det_boxes, torch.tensor(wheels_boxes).float()])
    # print(percent_boxes)

    # wheel classifer, According to the sum of number of pixels in the box
    # print(det_labels)

    one_mask = torch.zeros(decode_mask.shape)
    one_mask[decode_mask > 0] = 1
    for i, label in enumerate(det_labels):
        if label == 'wheel':
            box = percent_boxes[i] * decode_mask.shape[-1]
            box = box.tolist()  # [x1, y1, x2, y2]
            box = list(map(lambda x:int(x), box))
            # print(decode_mask.shape)

            cls = decode_mask.squeeze()[box[1]:box[3], box[0]:box[2]].cpu()
            one_cls = one_mask.squeeze()[box[1]:box[3], box[0]:box[2]]

            if one_cls.sum() < 30:
                continue
            elif (cls == 1).sum() > (cls == 2).sum():
                det_labels[i] = 'rear'
            elif (cls == 1).sum() <= (cls == 2).sum():
                det_labels[i] = 'front'

    del_index = []
    for i, label in enumerate(det_labels):
        if label == 'wheel':
            # del det_labels[i]
            del_index.append(i)
    # print(det_labels)

    pred_dict = {}
    pred_dict['labels'] = []
    pred_dict['boxes'] = []
    pred_dict['confidence'] = []

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image, pred_dict

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)

    # Suppress specific classes, if needed
    # print(label_color_map)
    for i in range(det_boxes.size(0)):
        if i in del_index:
            continue
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)

        pred_dict['labels'].append(det_labels[i])
        pred_dict['boxes'].append(box_location)
        if det_labels[i] in ['rear', 'front']:
            pred_dict['confidence'].append(0.9)
        else:
            pred_dict['confidence'].append(det_scores[0][i].item())

    del draw

    return annotated_image, pred_dict


def detect_flip(original_image_flip, original_image, pred_dict, min_score=0.1, max_overlap=0.45, top_k=200,
                Nt=0.3, theresh=0.14, suppress=None, vis=False):

    _, pred_dict_flip = detect(original_image_flip, min_score=min_score, max_overlap=max_overlap, top_k=top_k, vis=False)
    # print(pred_dict_flip, pred_dict)
    if len(pred_dict_flip['labels']) == 0:
        return original_image, pred_dict

    filp_boxes = np.array(pred_dict_flip['boxes'])
    filp_boxes[:, 0] = original_image_flip.width - np.array(pred_dict_flip['boxes'])[:, 0] - 1
    filp_boxes[:, 2] = original_image_flip.width - np.array(pred_dict_flip['boxes'])[:, 2] - 1
    filp_boxes = filp_boxes[:, [2, 1, 0, 3]]

    pred_dict_flip['boxes'] = filp_boxes.tolist()

    total_boxes = pred_dict['boxes'] + pred_dict_flip['boxes']
    total_confidence = pred_dict['confidence'] + pred_dict_flip['confidence']
    total_labels = pred_dict['labels'] + pred_dict_flip['labels']

    keep = py_cpu_softnms(np.array(total_boxes), np.array(total_confidence), Nt=Nt, thresh=theresh)
    # print(keep)

    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)

    total_pred_dict = {}
    total_pred_dict['boxes'] = []
    total_pred_dict['confidence'] = []
    total_pred_dict['labels'] = []
    for i in keep:
        # Boxes
        box_location = total_boxes[i]
        draw.rectangle(xy=box_location, outline=label_color_map[total_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            total_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(total_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[total_labels[i]])
        draw.text(xy=text_location, text=total_labels[i].upper(), fill='white',
                  font=font)

        total_pred_dict['boxes'].append([round(j) for j in box_location])
        total_pred_dict['confidence'].append(total_confidence[i])
        total_pred_dict['labels'].append(total_labels[i])

    return annotated_image, total_pred_dict


if __name__ == '__main__':
    from pprint import pprint

    use_flip = False

    img_path = '../data/sub_test/test/000000.jpg'
    # img_path = '../data/train/000000.jpg'
    # img_path = '../000874.jpg'
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    # original_image = transforms.RandomHorizontalFlip(1)(original_image)

    annotated_image, pred_dict = detect(original_image.copy(), min_score=0.1, max_overlap=0.45, top_k=200, vis=False)

    if use_flip:
        original_image_flip = transforms.RandomHorizontalFlip(1)(original_image)
        annotated_image, pred_dict = detect_flip(original_image_flip, original_image, pred_dict=pred_dict,
                                                       min_score=0.2, max_overlap=0.45, top_k=200)

    annotated_image.show()
    pprint(pred_dict)
