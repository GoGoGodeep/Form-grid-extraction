# -*- coding: utf-8 -*-
#############################################################################
# Copyright (c) 2022  - Shanghai Davis Tech, Inc.  All rights reserved
"""
文件名：PerTransformation.py
2023-07-27: 周科帆, Davy @Davis Tech
"""
import os
import cv2
import numpy as np
from CVBaseTool import cv_imsave, cv_imread
import json

def identifyLines(v_img, v_erode_scale, v_dilate_scale):
    """
     IdentifyLines(v_img, v_erode_scale, v_dilate_scale)
     .   @brief 通过膨胀、腐蚀的操作，获得竖线与横线，用于后续函数操作
     .   @param v_imagePath: 指定原始图像，如："img/card.png"
     .   @param v_erode_scale: 超参数，腐蚀线形宽度
     .   @param v_dilate_scale: 超参数，膨胀线形宽度
     .   @return r_dilated_col, r_dilated_row: 返回处理后的竖线与横线图，类型为ndarray
    """
    # img_h, img_w = v_img.shape[0], v_img.shape[1]
    # 灰度图片
    gray = cv2.cvtColor(v_img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=35, C=-5)
    rows, cols = binary.shape

    # 自适应获取核值
    # 腐蚀核：腐蚀核是一个小的平面区域或模板，通常是正方形或圆形，其中心位置对应于腐蚀操作的中心像素。腐蚀核决定了腐蚀操作将从每个像素周围考虑多少邻域信息。较小的腐蚀核会导致更强的腐蚀效果，更大的腐蚀核则会产生较轻的腐蚀作用。
    # 膨胀核：膨胀核与腐蚀核类似，是一个小的平面区域或模板，其形状可以是正方形或圆形。膨胀核决定了膨胀操作中每个像素周围要考虑多少邻域信息。较小的膨胀核会产生较小的膨胀效果，而较大的膨胀核会产生更强的膨胀作用。
    # 识别横线:
    t_erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // v_erode_scale, 1))
    t_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // v_dilate_scale, 1))

    eroded = cv2.erode(binary, t_erode_kernel, iterations=1)
    r_dilated_col = cv2.dilate(eroded, t_dilate_kernel, iterations=1)

    # 识别竖线：
    # scale = 40 # scale越大，越能检测出不存在的线
    t_erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // v_erode_scale))
    t_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // v_dilate_scale))

    eroded = cv2.erode(binary, t_erode_kernel, iterations=1)
    r_dilated_row = cv2.dilate(eroded, t_dilate_kernel, iterations=1)

    return r_dilated_col, r_dilated_row


def findContours_img(original_img, opened):
    contours, hierarchy = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(contours, key=cv2.contourArea, reverse=True)[0]  # 对轮廓列表 contours 进行排序，按照轮廓的面积进行降序排列。取排序后的第一个轮廓，即最大的轮廓。
    rect = cv2.minAreaRect(c)  # 计算最大轮廓的旋转包围盒，返回一个矩形的信息，包括中心点坐标、宽度、高度和旋转角度。
    box = np.intp(cv2.boxPoints(rect))  # 将包围盒转换为四个顶点坐标，返回一个四边形的四个顶点。
    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)
    return box, draw_img


def Perspective_transform(box, original_img):
    # 该函数输入参数为包围盒的四个顶点坐标box和原始图像original_img

    # 定义变量 index_min 和 index_max 为包围盒的顶点索引
    index_min = 0
    index_max = 0
    # mark_min 和 mark_max 为最小和最大距离的初始值
    mark_min = 100000000
    mark_max = 0

    for i, (x, y) in enumerate(box):
        length = x ** 2 + y ** 2
        if mark_min > length:
            index_min = i
            mark_min = length
        if length > mark_max:
            index_max = i
            mark_max = length

    mark_index = []
    index = [index_min, index_max]

    for i in range(0, 4):
        if i not in index:
            mark_index.append(i)

    if box[mark_index[0]][1] > box[mark_index[1]][1]:
        right = mark_index[1]
        left = mark_index[0]
    else:
        right = mark_index[0]
        left = mark_index[1]

    # 将包围盒的四个顶点坐标转换为列表形式
    max_points = box.tolist()

    # 计算目标图像的宽度和高度，通过计算right顶点和最小距离顶点之间的欧氏距离和right顶点和最大距离顶点之间的欧氏距离。
    image_W = pow(
        pow(max_points[right][0] - max_points[index_min][0], 2) +
        pow(max_points[right][1] - max_points[index_min][1], 2),
        0.5)
    image_H = pow(
        pow(max_points[right][0] - max_points[index_max][0], 2) +
        pow(max_points[right][1] - max_points[index_max][1], 2),
        0.5)

    pts_o = np.float32([
        max_points[right], max_points[index_min],
        max_points[left], max_points[index_max]
    ])  # 初始
    pts_d = np.float32([
        [int(image_W + 1), 0], [0, 0],
        [0, int(image_H + 1)], [int(image_W + 1), int(image_H + 1)]
    ])  # 目标

    M = cv2.getPerspectiveTransform(pts_o, pts_d)  # 变换矩阵
    result_img = cv2.warpPerspective(original_img, M, (int(image_W + 1), int(image_H + 1)))

    return result_img, pts_o, pts_d


def preProcess(v_imagePath, v_rightImagePath='recImg.png'):
    """
     preProcess(v_imagePath, v_rightImagePath) -> r_img
     .   @brief 返回图片中表格部分，并将其透视矫正变换
     .   @param v_imagePath: 指定原始图像，如："img/card.png"
     .   @param v_rightImagePath: 指定矫正后图片保存路径，如："img/result.png"
     .   @return r_img: 返回预处理后的表格图像；如果原始图像路径错误，返回-1
     .   @return r_pts_o: 返回透视变换的初始坐标
     .   @return r_pts_d: 返回透视变换的目标坐标
     """
    if not os.path.isfile(v_imagePath):
        r_img = -1
        return r_img

    v_img = cv_imread(v_imagePath, cv2.IMREAD_COLOR)

    dilated_col, dilated_row = identifyLines(v_img, v_erode_scale=20, v_dilate_scale=15)
    # 标识表格轮廓
    merge = cv2.add(dilated_col, dilated_row)
    _, binary = cv2.threshold(merge, 127, 255, cv2.THRESH_BINARY)

    box, img = findContours_img(v_img, binary)
    # 得到透视变换后的图片，透视变换的初始坐标，透视变换的目的坐标
    r_img, pts_o, pts_d = Perspective_transform(box, v_img)

    cv_imsave(r_img, v_rightImagePath)

    r_pts_o, r_pts_d = pts_o, pts_d

    return r_img, r_pts_o, r_pts_d


def Inverse_Perspective_transform(v_original_img, v_pts_o, v_pts_d):
    """
    Inverse_Perspective_transform(v_original_img, v_pts_o, v_pts_d)
    .   @brief 实现逆透视变换
    .   @param v_original_img: 透视变换前，即原始图片地址
    .   @param v_pts_o: 透视变换的原始坐标
    .   @param v_pts_d: 透视变换的目的坐标
    .   @return r_M: 返回逆透视变换矩阵
    """
    orImg = cv2.imread(v_original_img)
    # 读取图像的尺寸用于后续操作
    h, w, _ = orImg.shape
    # 逆变换矩阵
    M = cv2.getPerspectiveTransform(v_pts_d, v_pts_o)

    r_M = M

    return r_M


def coordinate_transformation(v_coordinate, v_M):
    """
    coordinate_transformation(v_coordinate, v_M)
    .   @brief 获得透视变换矩阵后，对坐标进行变换
    .   @param v_coordinate: 输入的坐标点
    .   @param v_M: 输入的逆透视变换矩阵
    .   @return r_coordinate_tran: 返回逆透视变换后的坐标
    """
    # 升维度,（x, y）——> (x, y, 1)
    z_value = 1
    coordinates = np.hstack((v_coordinate, z_value * np.ones((v_coordinate.shape[0], 1))))
    # 转置， 1*3 ——> 3*1
    coordinates_T = coordinates.T
    # 进行坐标变换：用原始点的齐次坐标 (x_h, y_h, 1) 与变换矩阵 M 相乘，得到变换后的齐次坐标 (x'_h, y'_h, w)，其中 w 是一个缩放因子。
    inverse_coordinates = np.dot(v_M, coordinates_T)
    # 将齐次坐标转换回笛卡尔坐标：将变换后的齐次坐标 (x'_h, y'_h, w) 转换回笛卡尔坐标 (x', y')
    x = inverse_coordinates[0] / inverse_coordinates[2]
    y = inverse_coordinates[1] / inverse_coordinates[2]

    r_coordinate_tran = [x[0], y[0]]

    return r_coordinate_tran


def getInverseImg(v_ori_img, v_per_json, v_inv_json="", v_inv_img=""):
    """
    getInverseImg(v_original_json, v_original_img, v_inverse_img)
    .   @brief 对透视变换后的坐标进行逆透视变换，并保存变换后的json文件与图片
    .   @param v_ori_img: 原始图片，未经过透视变换
    .   @param v_per_json: 透视变换后得到的Json文件
    .   @param v_inv_json: 逆透视变换后得到的Json文件
    .   @param v_inv_img: 逆透视变换后在原始图片上绘制结果
    """
    # 如果不指定地址，就借用原文件名字和地址
    if v_inv_json == "":
        x = os.path.splitext(v_per_json)
        v_inv_json = x[0] + '_inv.json'
    else:
        pass
    if v_inv_img == "":
        x = os.path.splitext(v_ori_img)
        v_inv_img = x[0] + '_inv.png'
    else:
        pass

    # 得到透视变换的初始坐标与目的坐标
    _, pts_o, pts_d = preProcess(v_ori_img)
    # 获得逆透视变换矩阵
    M = Inverse_Perspective_transform(v_ori_img, pts_o, pts_d)

    ori_img = cv2.imread(v_ori_img)
    table_number = 1

    with open(v_per_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        for i in range(len(data)):
            x1, y1 = data['方格{}'.format(i + 1)]['方格起始点1']
            x2, y2 = data['方格{}'.format(i + 1)]['方格起始点2']

            xy1 = np.array([[x1, y1]])
            xy2 = np.array([[x2, y2]])

            xy1_ = coordinate_transformation(xy1, M)
            xy2_ = coordinate_transformation(xy2, M)

            data['方格{}'.format(i + 1)]['方格起始点1'] = xy1_
            data['方格{}'.format(i + 1)]['方格起始点2'] = xy2_

            cv2.rectangle(ori_img, (int(xy1_[0]), int(xy1_[1])), (int(xy2_[0]), int(xy2_[1])), (0, 255, 0), 2)
            cv2.putText(ori_img, str(table_number), (int(xy1_[0]) + 20, int(xy1_[1]) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            table_number += 1
    f.close()
    with open(v_inv_json, 'w', encoding='utf-8') as w:
        json.dump(data, w, ensure_ascii=False, indent=4)
    w.close()

    cv_imsave(ori_img, v_inv_img)


if __name__ == '__main__':
    t_img_path = 'ori_img.jpg'
    t_per_json = ""

    getInverseImg(v_ori_img=t_img_path,
                  v_per_json=t_per_json)
