# coding:utf8
__first_version_author__ = 'tylin'
__second_version_author__ = 'wfnian'
# Interface for accessing the Microsoft COCO dataset.

# Microsoft COCO Toolbox.	  version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import json
import time
import shutil
import os
from collections import defaultdict
import json
from pathlib import Path


class COCO:
    def __init__(self, annotation_file=None, origin_img_dir=""):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.origin_dir = origin_img_dir
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()  # imgToAnns　一个图片对应多个注解(mask) 一个类别对应多个图片
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index　　  给图片->注解,类别->图片建立索引
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def build(self, tarDir=None, tarFile='./new.json', N=1000):

        load_json = {'images': [], 'annotations': [], 'categories': [], 'type': 'instances', "info": {"description": "This is stable 1.0 version of the 2014 MS COCO dataset.", "url": "http:\/\/mscoco.org", "version": "1.0", "year": 2014, "contributor": "Microsoft COCO group", "date_created": "2015-01-27 09:11:52.357475"}, "licenses": [{"url": "http:\/\/creativecommons.org\/licenses\/by-nc-sa\/2.0\/", "id": 1, "name": "Attribution-NonCommercial-ShareAlike License"}, {"url": "http:\/\/creativecommons.org\/licenses\/by-nc\/2.0\/", "id": 2, "name": "Attribution-NonCommercial License"}, {"url": "http:\/\/creativecommons.org\/licenses\/by-nc-nd\/2.0\/",
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              "id": 3, "name": "Attribution-NonCommercial-NoDerivs License"}, {"url": "http:\/\/creativecommons.org\/licenses\/by\/2.0\/", "id": 4, "name": "Attribution License"}, {"url": "http:\/\/creativecommons.org\/licenses\/by-sa\/2.0\/", "id": 5, "name": "Attribution-ShareAlike License"}, {"url": "http:\/\/creativecommons.org\/licenses\/by-nd\/2.0\/", "id": 6, "name": "Attribution-NoDerivs License"}, {"url": "http:\/\/flickr.com\/commons\/usage\/", "id": 7, "name": "No known copyright restrictions"}, {"url": "http:\/\/www.usa.gov\/copyright.shtml", "id": 8, "name": "United States Government Work"}]}
        if not Path(tarDir).exists():
            Path(tarDir).mkdir()

        for i in self.imgs:
            if(N == 0):
                break
            tic = time.time()
            img = self.imgs[i]
            load_json['images'].append(img)
            fname = os.path.join(tarDir, img['file_name'])
            anns = self.imgToAnns[img['id']]
            for ann in anns:
                load_json['annotations'].append(ann)
            if not os.path.exists(fname):
                shutil.copy(self.origin_dir+'/'+img['file_name'], tarDir)
            print('copy {}/{} images (t={:0.1f}s)'.format(i, N, time.time() - tic))
            N -= 1
        for i in self.cats:
            load_json['categories'].append(self.cats[i])
        with open(tarFile, 'w+') as f:
            json.dump(load_json, f, indent=4)



if __name__ == "__main__":
    # train_dataset split
    train_anno_path = [
        "../annotations/annotations/person_keypoints_train2017.json",
        "../annotations/annotations/instances_train2017.json",
        "../annotations/annotations/captions_train2017.json",
    ]
    train_img_path = "../train2017"
    save_path ='./tiny_coco'
    train_img_save_path = os.path.join(save_path, "train2017")
    train_anno_save_path = os.path.join(save_path, "annotations")

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(train_img_save_path):
        os.mkdir(train_img_save_path)
    if not os.path.exists(train_anno_save_path):
        os.mkdir(train_anno_save_path)

    for anno_path in train_anno_path:
        coco = COCO(anno_path,
                    origin_img_dir=train_img_path)               # 完整的coco数据集的图片和标注的路径
        anno_filename = os.path.basename(anno_path)
        anno_path = os.path.join(train_anno_save_path, anno_filename)
        coco.build(train_img_save_path, anno_path, 50)  # 保存图片路径

    # val dataset split
    val_anno_path = [
        "../annotations/annotations/person_keypoints_val2017.json",
        "../annotations/annotations/instances_val2017.json",
        "../annotations/annotations/captions_val2017.json",
    ]
    val_img_path = "../val2017"
    save_path ='./tiny_coco'
    val_img_save_path = os.path.join(save_path, "val2017")
    val_anno_save_path = os.path.join(save_path, "annotations")

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(val_img_save_path):
        os.mkdir(val_img_save_path)
    if not os.path.exists(val_anno_save_path):
        os.mkdir(val_anno_save_path)

    for anno_path in val_anno_path:
        coco = COCO(anno_path,
                    origin_img_dir=val_img_path)               # 完整的coco数据集的图片和标注的路径
        anno_filename = os.path.basename(anno_path)
        anno_path = os.path.join(val_anno_save_path, anno_filename)
        coco.build(val_img_save_path, anno_path, 50)  # 保存图片路径