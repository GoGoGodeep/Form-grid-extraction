# -*- coding: utf-8 -*-
#############################################################################
# Copyright (c) 2022  - Shanghai Davis Tech, Inc.  All rights reserved
"""
文件名：CVBaseTool.py
2022-11-13: Davy @Davis Tech
"""
import cv2
import numpy as np


def cv_imread(v_imagePath, v_flags=-1):
    """
     cv_imread(v_imagePath) -> r_img
     .   @brief 支持中文路径读取图片，返回ndarray数组
     .   @param v_imagePath: 指定原始图像，如："img/1.png"
     .   @param v_flags: 读取参数，如：1(cv2.IMREAD_COLOR，三通道)，-1,4(cv2.IMREAD_COLOR. 四通道)
     .   @return r_img: 返回图像ndarray数组
     """
    cv2.IMREAD_ANYCOLOR
    r_img = cv2.imdecode(np.fromfile(v_imagePath, dtype=np.uint8), v_flags)
    return r_img

def cv_imsave(v_img_ndarray, v_save_path, v_type='png'):
    """
     cv_imsave(v_img_ndarray, v_save_path, v_type='png')
     .   @brief 支持中文路径读取图片，返回ndarray数组
     .   @param v_img_ndarray: 待保存图像数组
     .   @param v_save_path: 图像保存路径
     """
    if v_type == 'png':
        v_txt = '.png'
    elif v_type == 'jpg':
        v_txt = '.jpg'
    else:
        v_txt = '.png'
    cv2.imencode(v_txt, v_img_ndarray)[1].tofile(v_save_path)


if __name__ == '__main__':
    tmp = cv_imread("中文img/test1.png")
    cv_imsave(tmp, '中文img/1.png')
    print(type(tmp))
