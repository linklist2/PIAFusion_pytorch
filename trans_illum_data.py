# -*- coding: utf-8 -*-
"""
@Time ： 2022/6/21 15:20
@Auth ： zxc(https://github.com/linklist2)
@File ：trans_illum_data.py
@IDE ：PyCharm
@Function ：将PIAFusion的data_MSRS.h5文件数据集转换为文件夹存放图片的形式
"""
import argparse
import os

import cv2
import h5py
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch PIAFusion')
    parser.add_argument('--h5_path', metavar='DIR', default='datasets/data_illum.h5',
                        help='path to dataset')  # PIAFusion tesnorflow版本提供的data_MSRS.h5文件路径
    parser.add_argument('--cls_root_path', metavar='DIR', default='datasets/cls_dataset',
                        help='path to dataset')  # 转换后的图片存储位置
    args = parser.parse_args()

    h5_path = args.h5_path  # PIAFusion tesnorflow版本提供的data_MSRS.h5文件路径
    cls_root_path = args.cls_root_path  # 转换后的图片存储位置

    f = h5py.File(h5_path, 'r')
    sources = f['data'][:]
    sources = np.transpose(sources, (0, 3, 2, 1))
    images = sources[:, :, :, 0:3]
    labels = sources[:, 0, 0, 3:5]
    # [0, 1]表示night, [1,0]表示day
    day_iter = 0
    night_iter = 0

    day_dir = os.path.join(cls_root_path, 'day')
    night_dir = os.path.join(cls_root_path, 'night')

    try:
        os.makedirs(night_dir)
        os.makedirs(day_dir)
    except:
        pass

    for image, label in tqdm(zip(images, labels), total=images.shape[0]):
        image = np.uint8(image * 255)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if (label == [0, 1]).all():
            cv2.imwrite(os.path.join(night_dir, f'night_{night_iter}.png'), image)
            night_iter += 1
        elif (label == [1, 0]).all():
            cv2.imwrite(os.path.join(day_dir, f'day_{day_iter}.png'), image)
            day_iter += 1
