# Here, we just provide the mode which tests the samples in a pointed directory.
# if You want to use the real-time camera for object detection, you can have a recommendation from egs/voc2007/faster_rcnn/test.py


import os
import cv2
import time
import random
import numpy as np

import torch
import torchvision
from torchvision.ops.boxes import batched_nms
import torchvision.transforms.functional as F

from pathlib import Path
from PIL import Image
from tqdm import tqdm

from model import DETR
from train import get_parser
from nets.backbone import build_backbone
from nets.transformer import build_transformer

import local.transforms as T

randomresize = T.RandomResize([800], max_size=1333)
# img = F.to_tensor(img)
normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)
 
 
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b
 
 
def filter_boxes(scores, boxes, confidence=0.7, apply_nms=True, iou=0.5):
    keep = scores.max(-1).values > confidence
    scores, boxes = scores[keep], boxes[keep]
 
    if apply_nms:
        top_scores, labels = scores.max(-1)
        keep = batched_nms(boxes, top_scores, labels, iou)
        scores, boxes = scores[keep], boxes[keep]
 
    return scores, boxes

def plot_one_box(x, img, color=None, label=None, line_thickness=1):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def main(args):
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    num_classes = 91
    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        device=device,
    )

    checkpoint = torch.load(args.pretrained_model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], False)

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("parameters:", n_parameters)
 
    image_Totensor = torchvision.transforms.ToTensor()
    image_file_path = os.listdir(args.test_samples_dir)
 
    for image_item in tqdm(image_file_path):
        print("inference_image:", image_item)
        image_path = os.path.join(args.test_samples_dir, image_item)
        image = Image.open(image_path)
        image_tensor = image_Totensor(image)
        # image_resize, _ = randomresize(image)
        # image_tensor = F.to_tensor(image_resize)
        # image_tensor, _ = normalize(image_tensor)
        image_tensor = torch.reshape(image_tensor, [-1, image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2]])
        image_tensor = image_tensor.to(device)
        time1 = time.time()
        inference_result = model(image_tensor)
        time2 = time.time()
        print("inference_time:", time2 - time1)
        probas = inference_result['pred_logits'].softmax(-1)[0, :, :-1].cpu()
        bboxes_scaled = rescale_bboxes(inference_result['pred_boxes'][0,].cpu(),
                                       (image_tensor.shape[3], image_tensor.shape[2]))
        scores, boxes = filter_boxes(probas, bboxes_scaled)
        scores = scores.data.numpy()
        boxes = boxes.data.numpy()
        for i in range(boxes.shape[0]):
            class_id = scores[i].argmax()
            label = CLASSES[class_id]
            confidence = scores[i].max()
            text = f"{label} {confidence:.3f}"
            image = np.array(image)
            plot_one_box(boxes[i], image, label=text)
            
        # On server, this imshow function can't be used.
        # cv2.imshow("images", cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        # cv2.waitKey()
        if boxes.shape[0] > 0:
            image = Image.fromarray(image)
        image.save(os.path.join(args.test_results_dir, image_item))

    ## test fps
    results = []
    test_interval = 100
    test_fps_sample = "samples/street.jpg"
    image = Image.open(image_path)
    image_tensor = image_Totensor(image)
    image_tensor = torch.reshape(
        image_tensor,
        [-1, image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2]]
        )
    image_tensor = image_tensor.to(device)
    t1 = time.time()
    for _ in range(test_interval):
        with torch.no_grad():
            inference_result = model(image_tensor)
            probas = inference_result['pred_logits'].softmax(-1)[0, :, :-1].cpu()
            bboxes_scaled = rescale_bboxes(inference_result['pred_boxes'][0,].cpu(),(image_tensor.shape[3], image_tensor.shape[2]))
        scores, boxes = filter_boxes(probas, bboxes_scaled)
        results.append([])
        scores = scores.data.numpy()
        boxes = boxes.data.numpy()
        for i in range(boxes.shape[0]):
            class_id = scores[i].argmax()
            label = CLASSES[class_id]
            confidence = scores[i].max()
            c_pred     = (boxes[i], confidence, label)

            results[-1].extend(c_pred)

    t2 = time.time()
    tact_time = (t2 - t1) / test_interval
    print(str(tact_time) + ' seconds, ' + str(1/tact_time) + ' FPS, @batch_size 1')
 
 
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.test_results_dir:
        Path(args.test_results_dir).mkdir(parents=True, exist_ok=True)
    main(args)