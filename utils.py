import matplotlib.pyplot as plt
import torch


def plot_img(img, cmap='gray'):
    plt.figure(figsize=(9, 16))
    if cmap != 'gray':
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.show()


def box_corner2center(box_corner):
    """
    :param box: x1, y1, x2, y2
    :return: box: centerX, centerY, h, w
    """
    x1, y1, x2, y2 = box_corner
    centerX = (x1 + x2) // 2
    centerY = (y1 + y2) // 2
    h = y2 - y1
    w = x2 - x1
    box_center = centerX, centerY, h, w
    return box_center


def box_center2corner(box_center):
    """
    :param box: centerX, centerY, h, w
    :return: box: x1, y1, x2, y2
    """
    centerX, centerY, h, w = box_center
    x1 = centerX - w // 2
    x2 = centerX + w // 2
    y1 = centerY - h // 2
    y2 = centerY + h // 2
    return x1, y1, x2, y2


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]


    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def iou_per_box(box_pred, box_label):
    """
    :param boxes_preds: [x1, y1, x2, y2]
    :param boxes_labels: [x1, y1, x2, y2]
    :return: iou
    """
    box1_x1, box1_y1, box1_x2, box1_y2 = box_pred
    box2_x1, box2_y1, box2_x2, box2_y2 = box_label

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))


    return intersection / (box1_area + box2_area - intersection + 1e-6)