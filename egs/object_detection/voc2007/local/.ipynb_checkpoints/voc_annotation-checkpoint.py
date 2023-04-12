import argparse
import os
import random
import xml.etree.ElementTree as ET

import numpy as np


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="/userhome/data/voc_2007_2012",
        help="Path for VOC_2007.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/voc2007",
        help="Path for output train and val txt files.",
    )
    
    parser.add_argument(
        "--class-txt",
        type=str,
        default="download/voc2007_model_data/voc_classes.txt",
        help="""Path for the classes txt files. \
        必须要修改，用于生成2007_train.txt、2007_val.txt的目标信息 \
        与训练和预测所用的classes_path一致即可 \
        如果生成的2007_train.txt里面没有目标信息 \
        那么就是因为classes没有设定正确 \
        仅在annotation_mode为0和2的时候有效""",
    )

    parser.add_argument(
        "--year",
        type=int,
        default=2007,
        help="The year id for using dataset, eg: 2007 or 2012.",
    )

    parser.add_argument(
        "--trainval-percent",
        type=float,
        default=0.9,
        help="""The percent for train and val, respectively to test. \ 
        trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1 \
        train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1 \
        仅在annotation_mode为0和1的时候有效""",
    )

    parser.add_argument(
        "--train-percent",
        type=float,
        default=0.9,
        help="The percent for train, respectively to train.",
    )
    
    parser.add_argument(
        "--annotation-mode",
        type=int,
        default=0,
        help="""annotation_mode用于指定该文件运行时计算的内容 \
        annotation_mode为0代表整个标签处理过程，包括获得VOCdevkit/VOC2007/ImageSets里面的txt以及训练用的2007_train.txt、2007_val.txt \
        annotation_mode为1代表获得VOCdevkit/VOC2007/ImageSets里面的txt \
        annotation_mode为2代表获得训练用的2007_train.txt、2007_val.txt""",
    )
    
    return parser
    
#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def convert_annotation(VOCdevkit_path, classes, nums, year, image_id, list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml'%(year, image_id)), encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        
        nums[classes.index(cls)] = nums[classes.index(cls)] + 1

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    VOCdevkit_path = args.data_dir
    out_dir = args.out_dir
    annotation_mode = args.annotation_mode
    year = args.year
    trainval_percent = args.trainval_percent
    train_percent = args.train_percent
    
    classes_path = args.class_txt
    classes, _      = get_classes(classes_path)
    
    VOCdevkit_sets  = [(str(year), 'train'), (str(year), 'val')]
    ## 统计数据量
    photo_nums  = np.zeros(len(VOCdevkit_sets))
    nums        = np.zeros(len(classes))
    
    if " " in os.path.abspath(VOCdevkit_path):
        raise ValueError("数据集存放的文件夹路径与图片名称中不可以存在空格，否则会影响正常的模型训练，请注意修改。")

    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate txt in ImageSets.")
        xmlfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/Annotations')
        saveBasePath    = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main')
        temp_xml        = os.listdir(xmlfilepath)
        total_xml       = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num     = len(total_xml)  
        list    = range(num)  
        tv      = int(num*trainval_percent)  
        tr      = int(tv*train_percent)  
        trainval= random.sample(list,tv)  
        train   = random.sample(trainval,tr)  
        
        print("train and val size",tv)
        print("train size",tr)
        ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
        ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
        ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
        fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
        
        for i in list:  
            name=total_xml[i][:-4]+'\n'  
            if i in trainval:  
                ftrainval.write(name)  
                if i in train:  
                    ftrain.write(name)  
                else:  
                    fval.write(name)  
            else:  
                ftest.write(name)  
        
        ftrainval.close()  
        ftrain.close()  
        fval.close()  
        ftest.close()
        print("Generate txt in ImageSets done.")

    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate 2007_train.txt and 2007_val.txt for train.")
        type_index = 0
        for year, image_set in VOCdevkit_sets:
            image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt'%(year, image_set)), encoding='utf-8').read().strip().split()
            list_file = open('%s/%s_%s.txt'%(out_dir, year, image_set), 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write('%s/VOC%s/JPEGImages/%s.jpg'%(os.path.abspath(VOCdevkit_path), year, image_id))

                convert_annotation(VOCdevkit_path, classes, nums, year, image_id, list_file)
                list_file.write('\n')
            photo_nums[type_index] = len(image_ids)
            type_index += 1
            list_file.close()
        print("Generate 2007_train.txt and 2007_val.txt for train done.")
        
        def printTable(List1, List2):
            for i in range(len(List1[0])):
                print("|", end=' ')
                for j in range(len(List1)):
                    print(List1[j][i].rjust(int(List2[j])), end=' ')
                    print("|", end=' ')
                print()

        str_nums = [str(int(x)) for x in nums]
        tableData = [
            classes, str_nums
        ]
        colWidths = [0]*len(tableData)
        len1 = 0
        for i in range(len(tableData)):
            for j in range(len(tableData[i])):
                if len(tableData[i][j]) > colWidths[i]:
                    colWidths[i] = len(tableData[i][j])
        printTable(tableData, colWidths)

        if photo_nums[0] <= 500:
            print("训练集数量小于500，属于较小的数据量，请注意设置较大的训练世代（Epoch）以满足足够的梯度下降次数（Step）。")

        if np.sum(nums) == 0:
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("（重要的事情说三遍）。")
  
if __name__ == "__main__":
    random.seed(0)
    main()