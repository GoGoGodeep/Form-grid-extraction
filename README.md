### 文件内容：
#### PerTransformation.py：表单方格提取算法实现
#### GetChecked.py：算法调用应用

————————————————————————————————————————————————————————————————————————————————————

#### 主要功能：

1. **图像预处理**：
   - **灰度化**：将彩色图像转换为灰度图像。
   - **二值化**：使用自适应阈值方法将灰度图像转换为二值图像。
   - **腐蚀和膨胀**：通过腐蚀和膨胀操作提取图像中的横线和竖线。

2. **表格检测**：
   - **轮廓检测**：检测二值图像中的表格轮廓。
   - **透视变换**：计算透视变换的初始和目标坐标，并对表格图像进行透视变换，将其矫正为矩形。

3. **逆透视变换**：
   - **逆透视变换矩阵计算**：根据原始图像和透视变换后的图像计算逆透视变换矩阵。
   - **坐标变换**：使用逆透视变换矩阵将透视变换后的坐标转换回原始图像的视角。

4. **勾选框检测**：
   - **黑色点比例计算**：根据模板图片的json文件中的勾选框坐标，在输入图片对应位置计算黑色点比例。
   - **json文件修改**：根据黑色点比例修改输入图片的json文件，标记勾选框的状态。

#### 使用方法：

1. **安装依赖**：
   ```bash
   pip install opencv-python numpy
   ```

2. **准备数据**：
   - 准备需要处理的图片和对应的json文件。
   - 确保图片和json文件的路径正确。

3. **调用函数**：
   ```python
   # 对图像进行预处理和透视变换
   result_img, pts_o, pts_d = preProcess("path_to_your_image.png")

   # 对透视变换后的图像进行逆透视变换
   inv_M = Inverse_Perspective_transform("path_to_original_image.png", pts_o, pts_d)

   # 根据模板json文件检测勾选框，并修改json文件
   getChecked("path_to_your_json_file.json", "path_to_template_json_file.json", "path_to_your_image.png", "path_to_template_image.png")
   ```
