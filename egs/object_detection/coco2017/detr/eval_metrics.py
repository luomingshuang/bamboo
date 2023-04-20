# Here, we can't know the test dataset's ground truth.
# So for testing metrics, we use validation for an evaluation.

import os
cwd = os.getcwd()                  ## get the current path

import sys
sys.path.append(cwd)               ## add local to package 

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

from local.utils_map import get_coco_map, get_map
import local.transforms as T

# using the api from coco
from pycocotools.coco import COCO

randomresize = T.RandomResize([800], max_size=1333)
# img = F.to_tensor(img)
normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        cv2.putText(img, label, (c1[0], c1[1] -2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def main(args):
    '''
    Recall和Precision不像AP是一个面积的概念，因此在门限值（Confidence）不同时，网络的Recall和Precision值是不同的。
    默认情况下，本代码计算的Recall和Precision代表的是当门限值（Confidence）为0.5时，所对应的Recall和Precision值。
    受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，这样才可以计算不同门限条件下的Recall和Precision值
    因此，本代码获得的map_out/detection-results/里面的txt的框的数量一般会比直接predict多一些，目的是列出所有可能的预测框，
    '''
    #--------------------------------------------------------------------------------------#
    #   MINOVERLAP用于指定想要获得的mAP0.x，mAP0.x的意义是什么请同学们百度一下。
    #   比如计算mAP0.75，可以设定MINOVERLAP = 0.75。
    #
    #   当某一预测框与真实框重合度大于MINOVERLAP时，该预测框被认为是正样本，否则为负样本。
    #   因此MINOVERLAP的值越大，预测框要预测的越准确才能被认为是正样本，此时算出来的mAP值越低，
    #--------------------------------------------------------------------------------------#
    MINOVERLAP      = 0.5

    #--------------------------------------------------------------------------------------#
    #   受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，这样才可以计算mAP
    #   因此，confidence的值应当设置的尽量小进而获得全部可能的预测框。
    #   
    #   该值一般不调整。因为计算mAP需要获得近乎所有的预测框，此处的confidence不能随便更改。
    #   想要获得不同门限值下的Recall和Precision值，请修改下方的score_threhold。
    #--------------------------------------------------------------------------------------#
    confidence      = 0.02

    #---------------------------------------------------------------------------------------------------------------#
    #   Recall和Precision不像AP是一个面积的概念，因此在门限值不同时，网络的Recall和Precision值是不同的。
    #   
    #   默认情况下，本代码计算的Recall和Precision代表的是当门限值为0.5（此处定义为score_threhold）时所对应的Recall和Precision值。
    #   因为计算mAP需要获得近乎所有的预测框，上面定义的confidence不能随便更改。
    #   这里专门定义一个score_threhold用于代表门限值，进而在计算mAP时找到门限值对应的Recall和Precision值。
    #---------------------------------------------------------------------------------------------------------------#
    score_threhold  = 0.5

    #-------------------------------------------------------#
    #   map_vis用于指定是否开启map计算的可视化
    #-------------------------------------------------------#
    map_vis         = True

    #-------------------------------------------------------#
    #   结果输出的文件夹，默认为map_out
    #-------------------------------------------------------#
    map_out_path    = 'detr/exp/map_out'

    test_mode = "val"

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-images')):
        os.makedirs(os.path.join(map_out_path, 'detection-images'))
    if not os.path.exists(os.path.join(map_out_path, 'groundtruth-images')):
        os.makedirs(os.path.join(map_out_path, 'groundtruth-images'))

    detection_images_dir = os.path.join(map_out_path, "detection-images")
    groundtruth_images_dir = os.path.join(map_out_path, "groundtruth-images")

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

    print("Start loading model weights...")

    checkpoint = torch.load(args.pretrained_model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], False)

    model.to(device)
    model.eval()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("parameters:", n_parameters)

    print("Finish loading model weights...")
 
    image_Totensor = torchvision.transforms.ToTensor()

    # you can change this discrete path according to yourself.
    test_dataset_dir = ""
    anno_file = ""
    if test_mode == "test":
        test_dataset_dir = "/home/bcxiong1/data/coco2017/images/test2017"
    if test_mode == "val":
        test_dataset_dir = "/home/bcxiong1/data/coco2017/images/val2017"
        ann_file = "/home/bcxiong1/data/coco2017/images/annotations/instances_val2017.json"
    
    image_file_path = os.listdir(test_dataset_dir)
 
    print("Get predict results.")
    for image_item in tqdm(image_file_path):
        image_path = os.path.join(test_dataset_dir, image_item)

        image_id = os.path.split(image_path)[-1][:-4]
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w")
        with torch.no_grad():
            image = Image.open(image_path)
            image_tensor = image_Totensor(image)
            if image_tensor.size(0) == 1:
                continue
            # print(image_tensor.shape)
            # image_resize, _ = randomresize(image) ## resize as training
            # image_tensor = F.to_tensor(image_resize)
            # image_tensor, _ = normalize(image_tensor) ## normalize as training
            image_tensor = torch.reshape(image_tensor,[-1, image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2]])
            image_tensor = image_tensor.to(device)
            time1 = time.time()
            inference_result = model(image_tensor)
            time2 = time.time()
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

                left, top, right, bottom = boxes[i]
                f.write("%s %s %s %s %s %s\n" % (label, str(confidence)[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

            if boxes.shape[0] != 0:
                image = Image.fromarray(image)
            # image.save(os.path.join(detection_images_dir, os.path.split(image_path)[-1]))

        f.close()
        # On server, this imshow function can't be used.
        # cv2.imshow("images", cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        # cv2.waitKey()
    print("Get predict results done.")

    print("Get ground truth results.")  ## just for val dataset
    coco = COCO(ann_file)
    for image_item in tqdm(image_file_path):
        image_path = os.path.join(test_dataset_dir, image_item)
        image_id = os.path.split(image_path)[-1][:-4]
        image_id_new = image_id[-6:]
        annids = coco.getAnnIds(imgIds=int(image_id_new), iscrowd=None)
        anns = coco.loadAnns(annids)
        # if image_id_new == "473237": print(anns)
        f = open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"),"w")
        image = Image.open(image_path)
        for i in range(len(anns)):
            bbox = anns[i]["bbox"]

            left, top, bottom, right = bbox[0], bbox[1], bbox[1]+bbox[3], bbox[0]+bbox[2]
            catid = anns[i]["category_id"]
            cat = coco.loadCats(catid)
            label = cat[0]["name"]

            image = np.array(image)
            bbox = [left, top, right, bottom]
            plot_one_box(bbox, image, label=f"{label}")

            f.write("%s %s %s %s %s\n" % (label, str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))
            if len(anns) != 0:
                image = Image.fromarray(image)
        # image.save(os.path.join(groundtruth_images_dir, os.path.split(image_path)[-1]))

        f.close()
        
    print("Get ground truth results done.")

    print("Get map.")
    get_map(MINOVERLAP, True, score_threhold=score_threhold, path=map_out_path)
    print("Get map done.")

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)