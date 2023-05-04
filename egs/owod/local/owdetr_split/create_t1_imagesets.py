from pycocotools.coco import COCO
import numpy as np

T1_CLASS_NAMES = [
    "airplane","bicycle","bird","boat","bus","car",
    "cat","cow","dog","horse","motorcycle","sheep",
    "train","elephant","bear","zebra","giraffe","truck","person"
]



# Train
coco_annotation_file = '/home/bcxiong1/data/coco2017/annotations/instances_train2017.json'
dest_file = '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR/VOC2007/ImageSets/Main/t1_train.txt'

coco_instance = COCO(coco_annotation_file)

image_ids = []
cls = []

for index, image_id in enumerate(coco_instance.imgToAnns):
    image_details = coco_instance.imgs[image_id]
    classes = [coco_instance.cats[annotation['category_id']]['name'] for annotation in coco_instance.imgToAnns[image_id]]

    if not set(classes).isdisjoint(T1_CLASS_NAMES):
        image_ids.append(image_details['file_name'].split('.')[0])
        cls.extend(classes)

(unique, counts) = np.unique(cls, return_counts=True)
print({x:y for x,y in zip(unique, counts)})

c = 0 
for x,y in zip(unique, counts):
    if x in T1_CLASS_NAMES:
        c += y
print(c)
print(len(image_ids))

with open(dest_file, 'w') as file:
    for image_id in image_ids:
        file.write(str(image_id)+'\n')

print('Created train file')

# Test
coco_annotation_file = '/home/bcxiong1/data/coco2017/annotations/instances_val2017.json'
dest_file = '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR/VOC2007/ImageSets/Main/t1_test.txt'

coco_instance = COCO(coco_annotation_file)

image_ids = []
cls = []
for index, image_id in enumerate(coco_instance.imgToAnns):
    image_details = coco_instance.imgs[image_id]
    classes = [coco_instance.cats[annotation['category_id']]['name'] for annotation in coco_instance.imgToAnns[image_id]]
    if not set(classes).isdisjoint(T1_CLASS_NAMES):
        image_ids.append(image_details['file_name'].split('.')[0])
        cls.extend(classes)

(unique, counts) = np.unique(cls, return_counts=True)
print({x:y for x,y in zip(unique, counts)})

c = 0
for x,y in zip(unique, counts):
    if x in T1_CLASS_NAMES:
        c += y
print(c)
print(len(image_ids))

with open(dest_file, 'w') as file:
    for image_id in image_ids:
        file.write(str(image_id)+'\n')
print('Created test file')
