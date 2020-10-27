#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 22:40:36 2020

@author: geoff
"""

import torch
import torch.nn as nn


def IoU(bboxes1, bboxes2, wh=False):
    iou = 0
    A = bboxes1.size(0)
    B = bboxes2.size(0)
    max_xy = torch.min(bboxes1[:, 2:].unsqueeze(1).expand(A, B, 2),
                       bboxes2[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(bboxes1[:, :2].unsqueeze(1).expand(A, B, 2),
                       bboxes2[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter_area = inter[:, :, 0] * inter[:, :, 1]
    
    area_a = ((bboxes1[:, 2]-bboxes1[:, 0]) *
              (bboxes1[:, 3]-bboxes1[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((bboxes2[:, 2]-bboxes2[:, 0]) *
              (bboxes2[:, 3]-bboxes2[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter_area / union

# https://zhuanlan.zhihu.com/p/94799295
def DIoU(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))
    if rows * cols == 0:#
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True
    # #xmin,ymin,xmax,ymax->[:,0],[:,1],[:,2],[:,3]
    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1] 
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]
    
    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2 
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2 
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:]) 
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2]) 
    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:]) 
    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious,min=-1.0,max = 1.0)
    if exchange:
        dious = dious.T
    return dious

if __name__=="__main__":
    bboxes1 = torch.randn(2,4)
    bboxes2 = torch.randn(2,4)
    iou = IoU(bboxes1, bboxes2)
    dious = DIoU(bboxes1, bboxes2)
    
    print("iou:", iou, dious)