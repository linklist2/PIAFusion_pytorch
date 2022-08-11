# -*- coding: utf-8 -*-
"""
@Time ： 2022/7/20 13:51
@Auth ： zxc (https://github.com/linklist2)
@File ：plotRegionZoom.py
@IDE ：PyCharm
@Function ：
"""

import cv2
import numpy as np


def labelImg(img_path: str, label: list, save=False, color='red', line_width=2):
    """
    对图像绘制矩形框，并提取矩形框内的像素值

    :param img_path: 图片路径
    :param label: 矩形框的坐标, [x_min, y_min,x_max,y_max], 横轴是x, 纵轴是y
    :param save: 是否保存带有矩形框的图片和区域图片
    :param color: 矩形框的颜色
    :param line_width: 矩形框线条的宽度
    :return: 带有矩形框的图片数组，矩形框内的像素值（带有矩形框）
    """

    b, g, r = 0, 0, 255
    if isinstance(color, str):
        if color == 'red':
            b, g, r = 0, 0, 255
        elif color == 'green':
            b, g, r = 0, 255, 0
        elif color == 'blue':
            b, g, r = 255, 0, 0
    else:
        b, g, r = color

    x_min, y_min, x_max, y_max = label
    img_format = img_path.split('.')[-1]  # 读取图片的格式，e.g. png
    img = cv2.imread(img_path)

    # 读取矩形框内的像素值
    # numpy格式中第一个维度是y, 第二个维度是x
    region = img[y_min:y_max + 1, x_min:x_max + 1]

    # 存储该区域内的像素值至一个新的图片文件中

    # 绘制矩形框
    pt1 = (x_min, y_min)
    pt2 = (x_max, y_max)
    cv2.rectangle(img, pt2, pt1, (b, g, r), line_width)  # cv2 里也是先x，后y，x轴沿水平方向向右，y轴沿垂直方向向下

    if save:
        # 若下一行写在cv2.rectangle前，则保存的区域图像不带矩形框；写在这里则会带着框，因为python对象的机制，region一并发生了变化。
        cv2.imwrite(f'region.{img_format}', region)
        cv2.imwrite(f'rectangle.{img_format}', img)

    return img, region


# np.array


def zoomPlot(img: np.ndarray, region: np.ndarray, plot_place: str, scale: float, save_path: str):
    """
    生成局部区域放大图

    :param img: 带有矩形框的大图
    :param region: 截取的区域
    :param plot_place: 放大后的区域部分放置在大图中的位置， 可选有'top left', 'top right', 'upper left', 'upper right'
    :param scale: 区域放大的倍数
    :param save_path: 保存图片的路径
    :return: 局部区域放大图
    """

    o_h, o_w = img.shape[:2]
    # 放大
    region = cv2.resize(region, (0, 0), fx=scale, fy=scale)
    h, w = region.shape[:2]

    # 替换大图中的对应区域
    if plot_place == 'top left':  # 左上
        img[0:h, 0:w] = region
    elif plot_place == 'top right': # 右上
        img[0:h, o_w - w:] = region
    elif plot_place == 'bottom left':  # 左下
        img[o_h - h:, 0:w] = region
    elif plot_place == 'bottom right':  # 右下
        img[o_h - h:, o_w - w:] = region

    cv2.imwrite(save_path, img)


def plotMultiRegion(img_path: str, region_list: list, color_list: list, line_width_list: list, save_path: str,
                    place_list: list, scale_list: list, zoom_bool:list):
    """
    绘制多个区域的放大效果

    :param img_path: 图片路径
    :param region_list: 多个区域的坐标信息， [[x_min, y_min,x_max,y_max], [x_min, y_min,x_max,y_max]], 横轴是x, 纵轴是y
    :param color_list: 多个区域框的颜色列表
    :param line_width_list: 线宽列表
    :param save_path: 存储路径
    :param place_list: 放大的区域放置在原图中什么位置列表 e.g. ['top right', 'upper right']
    :param scale_list: 缩放列表
    :param zoom_bool: 是否进行缩放
    :return: 带有多个区域放大效果的图片
    """

    assert len(region_list) == len(color_list) == len(line_width_list) == len(place_list) == len(scale_list)
    region_num = len(color_list)

    for index in range(region_num):
        if index == 0:
            path = img_path
        else:
            path = save_path
        img, region = labelImg(path, region_list[index], color=color_list[index],
                               line_width=line_width_list[index])
        if zoom_bool[index]:
            zoomPlot(img, region, scale=scale_list[index], plot_place=place_list[index], save_path=save_path)

        else:
            cv2.imwrite(save_path, img)


if __name__ == '__main__':
    region_list = [
        [137, 105, 167, 161],
        [149, 171, 172, 212]
    ]

    zoom_bool = [True, True]

    color_list = ['red', 'green']
    line_list = [1, 1]
    scale_list = [2, 2]
    place_list = ['top left', 'bottom right']
    plotMultiRegion('39.bmp', region_list=region_list, line_width_list=line_list, color_list=color_list,
                    place_list=place_list, scale_list=scale_list, save_path='multiregion.bmp', zoom_bool=zoom_bool)
