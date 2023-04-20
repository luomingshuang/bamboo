from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab,json

"""
以list形式存放的图片标注的字典，字典键值为{“image_id”,"category_id",“bbox”,“score”}

[{"image_id":42,"category_id":18,"bbox":[258.15,41.29,348.26,243.78],"score":0.236},
{"image_id":73,"category_id":11,"bbox":[61,22.75,504,609.67],"score":0.318},
{"image_id":73,"category_id":4,"bbox":[12.66,3.32,268.6,271.91],"score":0.726},
{"image_id":74,"category_id":18,"bbox":[87.87,276.25,296.42,103.18],"score":0.546},
{"image_id":74,"category_id":2,"bbox":[0,3.66,142.15,312.4],"score":0.3},
{"image_id":74,"category_id":1,"bbox":[296.55,93.96,18.42,58.83],"score":0.407},
{"image_id":74,"category_id":1,"bbox":[328.94,97.05,13.55,25.93],"score":0.611}]

Maybe you can have a look at:
https://zhuanlan.zhihu.com/p/134229574
and 
https://zhuanlan.zhihu.com/p/134236324

How to use MS COCO api:
https://github.com/chenjie04/Learning_the_COCO
and
https://blog.csdn.net/u013085021/article/details/105905306

You also can learn from this url:
https://blog.csdn.net/weixin_43805402/article/details/120452972  <----重要

Annotation file: ann.json
{
    "images":[{"id": 73}],
    "annotations":[{
        "image_id":73,
        "category_id":1,
        "bbox":[10,10,50,100],
        "id":1,
        "iscrowd": 0,
        "area": 10
        }],
    "categories": [
        {"id": 1, "name": "person"}, 
        {"id": 2, "name": "bicycle"}, 
        {"id": 3, "name": "car"}
    ]
}

Result file: res.json
[{
    "image_id":73,
    "category_id":1,
    "bbox":[10,10,50,100],
    "score":0.9
}]

"""

ann_file = "/home/bcxiong1/data/coco2017/images/annotations/instances_val2017.json"
predict_file = "detr/exp/detections_val_detr-resnet50_predict_results.json"
gt_file = "detr/exp/detections_val_detr-resnet50_gt_results.json"

if __name__ == "__main__":
    cocoGt = COCO(ann_file)           # 标注文件的路径及文件名，json文件形式
    cocoDt = cocoGt.loadRes(predict_file)   # 自己的生成的结果的路径及文件名，json文件形式
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox") # annotype: segm, bbox, keypoints
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

       
    cocoGt = COCO(gt_file)
    cocoDt = cocoGt.loadRes(predict_file)   # 自己的生成的结果的路径及文件名，json文件形式
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox") # annotype: segm, bbox, keypoints
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()