
# 📑 表单结构化提取系统（Form Grid Extraction）

基于OpenCV实现合同等表单图像的**表格检测定位**与**勾选框识别**系统

![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-green) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

## 🛠️ 核心功能

### 🔍 图像预处理管线
- **🖼️ 灰度化**：RGB图像转灰度空间
- **⚫️ 自适应二值化**：动态阈值分割文本区域
- **⛏️ 形态学操作**：腐蚀膨胀联合提取表格线特征

### 📊 表格结构解析
- **📐 透视矫正**：自动检测表格轮廓并计算变换矩阵
- **🔄 逆透视映射**：保持原始坐标系的几何关系
- **📍 坐标解析**：输出表格单元格的精确坐标矩阵

### ✅ 勾选框识别
- **🎯 ROI定位**：基于模板匹配的复选框定位
- **🖤 像素密度分析**：计算黑色像素占比判断勾选状态
- **📝 JSON联动**：自动更新表单数据文件

---

## 📂 代码架构
```bash
├── PerTransformation.py    # 📄 核心算法实现（预处理/表格检测/坐标变换）
├── GetChecked.py           # 📄 应用层接口（勾选框检测与数据更新）
```

## 🚀 快速开始

### 环境配置
```bash
pip install opencv-python numpy
```

### 示例流程
```python
# 1. 表格结构提取
result_img, src_pts, dst_pts = preProcess("contract.png")

# 2. 建立坐标映射关系
inv_matrix = Inverse_Perspective_transform("original.png", src_pts, dst_pts)

# 3. 勾选框状态检测
getChecked("user_data.json", "template.json", "user_img.png", "template_img.png")
```
