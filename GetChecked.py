# -*- coding: utf-8 -*-
#############################################################################
# Copyright (c) 2023  - Shanghai Davis Tech, Inc.  All rights reserved
"""
文件名: GetCheckedPos.py
说明: 空白表格模板与勾选后的表格图片进行取差，获得差异像素点，得到对应的坐标，并把每个中心点（对应一个轮廓）的黑色点占所有点的比例统计出来
2023-08-01: 周科帆
"""
from CVBaseTool import cv_imsave, cv_imread
import cv2
import json


def getChecked(v_json, v_template_json, v_img, v_template_img):
    """
    getChecked(v_json, v_template_json, v_img, v_template_img)
    .   @brief 根据模版图片的json中的勾选框坐标，在输入图片对应位置计算黑色点比例，并修改json文件
    .   @param v_json: 输入图片json
    .   @param v_template_json: 模版图片json用于提供坐标
    .   @param v_img: 输入图片地址
    .   @param v_template_img: 模版图片地址
    .   @return: v_json: 返回json名称
    """
    img = cv_imread(v_img)
    img_template = cv_imread(v_template_img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=35, C=-5)

    with open(v_template_json, 'r', encoding='utf-8') as template_f:
        template_f_data = json.load(template_f)
    template_f.close()

    with open(v_json, 'r', encoding='utf-8') as f:
        f_data = json.load(f)
    f.close()

    # 遍历整个json
    for i in range(len(template_f_data)):
        f_data['方格{}'.format(i + 1)]['内容'] = []
        # 如果内容不为空继续
        if template_f_data['方格{}'.format(i + 1)]["内容"]:
            # 获取模版和输入图片的最外层表格的左上角坐标
            template_ori_x1, template_ori_y1 = template_f_data['方格{}'.format(i + 1)]["方格起始点1"]
            template_ori_x2, template_ori_y2 = template_f_data['方格{}'.format(i + 1)]["方格起始点2"]
            ori_x1, ori_y1 = f_data['方格{}'.format(i + 1)]["方格起始点1"]
            ori_x2, ori_y2 = f_data['方格{}'.format(i + 1)]["方格起始点2"]

            w2 = template_ori_x2 - template_ori_x1
            w1 = ori_x2 - ori_x1
            h2 = template_ori_y2 - template_ori_y1
            h1 = ori_y2 - ori_y1

            content = template_f_data['方格{}'.format(i + 1)]["内容"]
            # 遍历勾选框
            for j in content:
                template_x1, template_y1 = j["方格起始点1"]
                template_x2, template_y2 = j["方格起始点2"]

                # 计算模版图片中的偏移坐标
                relative_x1 = template_x1 - template_ori_x1
                relative_y1 = template_y1 - template_ori_y1
                relative_x2 = template_x2 - template_ori_x1
                relative_y2 = template_y2 - template_ori_y1

                # 根据模版图片的偏移坐标对输入图片的坐标进行偏移计算
                x1 = int(ori_x1 + relative_x1 * (w1 / w2))
                y1 = int(ori_y1 + relative_y1 * (h1 / h2))
                x2 = int(ori_x1 + relative_x2 * (w1 / w2))
                y2 = int(ori_y1 + relative_y2 * (h1 / h2))

                width = int(x2 - x1)
                height = int(y2 - y1)
                black_point = 0

                # 先遍历宽，再遍历高，将轮廓中的所有像素点遍历出来
                for wid in range(width):
                    for hei in range(height):
                        if binary[y1 + hei][x1 + wid] == 255:
                            black_point += 1
                proportion = black_point / (width * height)

                # print('方格{}'.format(i + 1))
                print(proportion)

                if proportion > 0.3:
                    f_data['方格{}'.format(i + 1)]['内容'].append(
                        {
                            '方格起始点1': [x1, y1],
                            '方格起始点2': [x2, y2],
                            '勾选状态': True,
                            '文本内容': j['文本内容'],
                            "勾选位置": [int((x1 + x2)/2), int((y1 + y2)/2)],
                            "f_value": proportion
                        }
                    )
                else:
                    f_data['方格{}'.format(i + 1)]['内容'].append(
                        {
                            '方格起始点1': [x1, y1],
                            '方格起始点2': [x2, y2],
                            '勾选状态': False,
                            '文本内容': j['文本内容'],
                            "勾选位置": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                            "f_value": proportion
                        }
                    )

    with open(v_json, 'w', encoding='utf-8') as w:
        json.dump(f_data, w, ensure_ascii=False, indent=4)
    w.close()

    return v_json


if __name__ == '__main__':
    t_img = "3.png"
    t_json = "3.json"
    t_template_json = "template.json"
    t_template_img = "template.png"

    getChecked(v_json=t_json,
               v_template_json=t_template_json,
               v_img=t_img,
               v_template_img=t_template_img)
