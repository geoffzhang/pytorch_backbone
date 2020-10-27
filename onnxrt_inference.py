#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/8/8 2:06 下午
# @Author : Xintao
# @File : onnxrt_inference.py
import numpy as np
import onnxruntime
import time
import cv2


onnx_model_path = "./output/pfld_gray.onnx"
img_path = "/home/geoff/workspace/github/face_landmark/test_data/1.png"
img = cv2.imread(img_path, 0)
show_img = True

# 网络输入是BGR格式的图片
img1 = cv2.resize(img, (112, 112))
#image_data = img1[np.newaxis].astype(np.float32) / 255 * 0 + 1 
image_data = np.random.rand(1,1,112,112).astype(np.float32) * 0+1


session = onnxruntime.InferenceSession(onnx_model_path, None)
# get the name of the first input of the model
input_name = session.get_inputs()[0].name
output = session.run([], {input_name: image_data})[1]
print(output)
