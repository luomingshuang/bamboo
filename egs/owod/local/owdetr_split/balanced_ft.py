import itertools
import random
import os
import xml.etree.ElementTree as ET
from fvcore.common.file_io import PathManager

from collections import deque
import numpy as np

class Store:
    def __init__(self, total_num_classes, items_per_class, shuffle=False):
        self.shuffle = shuffle
        self.items_per_class = items_per_class
        self.total_num_classes = total_num_classes
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]

    def add(self, items, class_ids):
        for idx, class_id in enumerate(class_ids):
            self.store[class_id].append(items[idx])

    def retrieve(self, class_id):
        if class_id != -1:
            items = []
            for item in self.store[class_id]:
                items.extend(item)
            if self.shuffle:
                random.shuffle(items)
            return items
        else:
            all_items = []
            for i in range(self.total_num_classes):
                items = []
                for item in self.store[i]:
                    items.append(item)
                all_items.append(items)
            return all_items

    def reset(self):
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]

    def __str__(self):
        s = self.__class__.__name__ + '('
        for idx, item in enumerate(self.store):
            s += '\n Class ' + str(idx) + ' --> ' + str(len(list(item))) + ' items'
        s = s + ' )'
        return s

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return sum([len(s) for s in self.store])

T1_CLASS_NAMES = [
    "airplane","bicycle","bird","boat","bus","car",
    "cat","cow","dog","horse","motorcycle","sheep",
    "train","elephant","bear","zebra","giraffe","truck","person"
]

T2_CLASS_NAMES = [ 
    "traffic light","fire hydrant","stop sign",
    "parking meter","bench","chair","dining table",
    "potted plant","backpack","umbrella","handbag","tie",
    "suitcase","microwave","oven","toaster","sink","refrigerator","bed","toilet","couch"
]

T3_CLASS_NAMES = [
    "frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","banana","apple","sandwich","orange","broccoli",
    "carrot","hot dog","pizza","donut","cake"
]

T4_CLASS_NAMES = [
    "laptop","mouse","remote","keyboard","cell phone",
    "book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush","wine glass","cup","fork","knife","spoon","bowl","tv","bottle"
]

UNK_CLASS = ["unknown"]

items_per_class = 100
annotation_location = '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR/VOC2007/Annotations'

# Change this accodingly for each task t*

### generate finetune data for task 2
# known_classes = list(itertools.chain(T1_CLASS_NAMES, T2_CLASS_NAMES))
# train_files = ['/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR/VOC2007/ImageSets/Main/t2_train.txt',
#                '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR/VOC2007/ImageSets/Main/t1_train.txt']
# dest_file = '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR/VOC2007/ImageSets/Main/t2_ft_' + str(items_per_class) + '.txt'

### generate finetune data for task 3
# known_classes = list(itertools.chain(T1_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES))
# train_files = ['/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR/VOC2007/ImageSets/Main/t3_train.txt',
#                '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR/VOC2007/ImageSets/Main/t2_train.txt',
#                '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR/VOC2007/ImageSets/Main/t1_train.txt']
# dest_file = '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR/VOC2007/ImageSets/Main/t3_ft_' + str(items_per_class) + '.txt'

### generate finetune data for task 4
known_classes = list(itertools.chain(T1_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES))
train_files = ['/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR/VOC2007/ImageSets/Main/t4_train.txt',
               '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR/VOC2007/ImageSets/Main/t3_train.txt',
               '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR/VOC2007/ImageSets/Main/t2_train.txt',
               '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR/VOC2007/ImageSets/Main/t1_train.txt']
dest_file = '/home/bcxiong1/codes/bamboo-mscoco2017-owod/egs/owod/data/OWDETR/VOC2007/ImageSets/Main/t4_ft_' + str(items_per_class) + '.txt'

file_names = []
for tf in train_files:
    with open(tf, mode="r") as myFile:
        file_names.extend(myFile.readlines())

random.shuffle(file_names)

image_store = Store(len(known_classes), items_per_class)

current_min_item_count = 0

for fileid in file_names:
    fileid = fileid.strip()
    anno_file = os.path.join(annotation_location, fileid + ".xml")

    with PathManager.open(anno_file) as f:
        tree = ET.parse(f)

    for obj in tree.findall("object"):
        cls = obj.find("name").text
        if cls in known_classes:
            image_store.add((fileid,), (known_classes.index(cls),))

    current_min_item_count = min([len(items) for items in image_store.retrieve(-1)])
    print(current_min_item_count)
    if current_min_item_count == items_per_class:
        break

filtered_file_names = []
for items in image_store.retrieve(-1):
    filtered_file_names.extend(items)

print(image_store)
print(len(filtered_file_names))
print(len(set(filtered_file_names)))

filtered_file_names = set(filtered_file_names)
filtered_file_names = map(lambda x: x + '\n', filtered_file_names)

with open(dest_file, mode="w") as myFile:
    myFile.writelines(filtered_file_names)

print('Saved to file: ' + dest_file)
