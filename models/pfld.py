#!/usr/bin/env python3
# -*- coding:utf-8 -*-

######################################################
#
# toplus.py -
# written by  zhangjianfeng
#
######################################################

import torch
import torch.nn as nn
import math
import os
width_multiplier = 0.25
res_multiplier = 1

def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inp * expand_ratio,
                inp * expand_ratio,
                3,
                stride,
                1,
                groups=inp * expand_ratio,
                bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PFLDInference(nn.Module):
    def __init__(self, num_output_channels):
        """       
        t   c    n   s   Operator                Input
        -   64   1   2   Conv3 × 3               112 × 112 × 3 
        -   64   1   1   Depthwise Conv3 × 3     56 × 56 × 64           
        2   64   5   2   Bottleneck              56 × 56 × 64  
        2   128  1   2   Bottleneck              28 × 28 × 64   
        4   128  6   1   Bottleneck              14 × 14 × 128   
        2   16   1   1   Bottleneck              14 × 14 × 128  
        -   32   1   2   Conv3 × 3               (S1) 14 × 14 × 16 
        -   128  1   1   Conv7 × 7               (S2) 7 × 2 × 32  
        -   128  1   -   -                       (S3) 1 × 1 × 128                    
        -   136  1   -   Full Connection         S1, S2, S3    
        """
        super(PFLDInference, self).__init__()
        self.display_network=1
        self.num_output_channels = num_output_channels
        self.conv1 = nn.Conv2d(3, int(64*width_multiplier), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(64*width_multiplier))
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(int(64*width_multiplier), int(64*width_multiplier), kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv2 = nn.Conv2d(int(64*width_multiplier), int(64*width_multiplier), kernel_size=3, stride=1, padding=1, groups=int(64*width_multiplier), bias=False)   # DWConv
        self.bn2 = nn.BatchNorm2d(int(64*width_multiplier))
        self.relu = nn.ReLU(inplace=True)

        self.conv3_1 = InvertedResidual(int(64*width_multiplier), int(64*width_multiplier), 2, False, 2)

        self.block3_2 = InvertedResidual(int(64*width_multiplier), int(64*width_multiplier), 1, True, 2)
        self.block3_3 = InvertedResidual(int(64*width_multiplier), int(64*width_multiplier), 1, True, 2)
        self.block3_4 = InvertedResidual(int(64*width_multiplier), int(64*width_multiplier), 1, True, 2)
        self.block3_5 = InvertedResidual(int(64*width_multiplier), int(64*width_multiplier), 1, True, 2)

        self.conv4_1 = InvertedResidual(int(64*width_multiplier), int(128*width_multiplier), 2, False, 2)

        self.conv5_1  = InvertedResidual(int(128*width_multiplier), int(128*width_multiplier), 1, False, 4)
        self.block5_2 = InvertedResidual(int(128*width_multiplier), int(128*width_multiplier), 1, True, 4)
        self.block5_3 = InvertedResidual(int(128*width_multiplier), int(128*width_multiplier), 1, True, 4)
        self.block5_4 = InvertedResidual(int(128*width_multiplier), int(128*width_multiplier), 1, True, 4)
        self.block5_5 = InvertedResidual(int(128*width_multiplier), int(128*width_multiplier), 1, True, 4)
        self.block5_6 = InvertedResidual(int(128*width_multiplier), int(128*width_multiplier), 1, True, 4)

        self.conv6_1 = InvertedResidual(int(128*width_multiplier), int(16*width_multiplier), 1, False, 2)  # [16*width_multiplier, 14, 14]

        self.conv7 = conv_bn(int(16*width_multiplier), int(32*width_multiplier), 3, 2, 1)  # [32*width_multiplier, 7, 7]
        # self.conv8 = conv_bn(int(32*width_multiplier), int(128*width_multiplier), math.ceil(7*res_multiplier), 1, 0)  # [128*width_multiplier, 1, 1]
        self.conv8 = nn.Conv2d(int(32*width_multiplier), int(128*width_multiplier), math.ceil(7*res_multiplier), 1, 0) 
        self.bn8 = nn.BatchNorm2d(int(128*width_multiplier))

        self.avg_pool1 = nn.AvgPool2d(int(14*res_multiplier))
        self.avg_pool2 = nn.AvgPool2d(math.ceil(7*res_multiplier))
        self.fc = nn.Linear(int(176*width_multiplier), self.num_output_channels)
    
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):  # x: 3, 112, 112
        if self.display_network:
            print("PFLDInterence:")
            print(x.shape)
        x = self.relu(self.bn1(self.conv1(x)))  # [64, 56, 56]
        if self.display_network:
            print(x.shape)
        x = self.relu(self.bn2(self.conv2(x)))  # [64, 56, 56]
        # x = self.relu(self.conv1(x))  # [64, 56, 56]
        # x = self.relu(self.conv2(x))  # [64, 56, 56]
        if self.display_network:
            print(x.shape)
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        out1 = self.block3_5(x)
        if self.display_network:
            print(out1.shape)
        x = self.conv4_1(out1)
        if self.display_network:
            print(x.shape)
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.block5_5(x)
        x = self.block5_6(x)
        if self.display_network:
            print(x.shape)
        x = self.conv6_1(x)
        if self.display_network:
            print(x.shape)
        x1 = self.avg_pool1(x)
        
        x1 = x1.view(x1.size(0), -1)

        x = self.conv7(x)
        if self.display_network:
            print(x.shape)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)

        # x3 = self.conv8(x)
        x3 = self.relu(self.conv8(x))
        if self.display_network:
            print(x3.shape)
        x3 = x3.view(x1.size(0), -1)
        
        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)

        self.display_network = 0
        return out1, landmarks


class AuxiliaryNet(nn.Module):
    def __init__(self):
        """
        Input           Operator           c     s                                
        28 × 28 × 64    Conv3 × 3          128   2                              
        14 × 14 × 128   Conv3 × 3          128   1                               
        14 × 14 × 128   Conv3 × 3          32    2                              
        7 × 7 × 32      Conv7 × 7          128   1                             
        1 × 1 × 128     Full Connection    32    1                                   
        1 × 1 × 32      Full Connection    3     -                                                                                                           
        """
        super(AuxiliaryNet, self).__init__()
        self.display_network=1
        self.conv1 = conv_bn(int(64*width_multiplier), int(128), 3, 2)
        self.conv2 = conv_bn(int(128), int(128), 3, 1)
        self.conv3 = conv_bn(int(128), int(32), 3, 2)
        self.conv4 = conv_bn(int(32),  int(128), int(7*res_multiplier), 1)
        self.max_pool1 = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(int(128), int(32))
        self.fc2 = nn.Linear(int(32), 3)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.display_network:
            print(x.shape)
        x = self.conv1(x)
        if self.display_network:
            print(x.shape)
        x = self.conv2(x)
        if self.display_network:
            print(x.shape)
        x = self.conv3(x)
        if self.display_network:
            print(x.shape)
        x = self.conv4(x)

        if self.display_network:
            print(x.shape)        
        x = self.max_pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.display_network:
            print(x.shape)        
        x = self.fc2(x)
        if self.display_network:
            print(x.shape)
        
        self.display_network = 0
        return x



    import sys
    sys.path.append("/home/geoff/workspace/github/framework/pytorch_backbone")
    from utils import utils_base as utils
    import pfld_no_bn
    from utils import fusion
    import collections
    input = torch.randn(1, 1, 112, 112)*0+1
    
    # 加载原始网络结构和模型
    pfld_backbone = PFLDInference(136)
    model_state_dict = torch.load("../pretrained_models/0_1_best.pth.tar",map_location=torch.device('cpu'))
    pfld_backbone.load_state_dict(model_state_dict["plfd_backbone"])
    pfld_backbone.eval()   # 设置为eval模式
    
    # 加载无bn网络结构
    new_pfld_backbone = pfld_no_bn.PFLDInference(136)
    new_pfld_backbone.eval()  # 设置为eval模式

    # 合并bn
    pre_module_v = None
    pre_module_name = None
    new_modules = collections.OrderedDict()
    new_modules_list = []
    new_modules_name_list = []
    last_module_v = None
    last_module_name = None
    for k, v in pfld_backbone.named_modules():
        if isinstance(v, nn.Conv2d) or isinstance(v, nn.Linear) or isinstance(v, nn.BatchNorm2d):
            pre_module_v = v if pre_module_v is None else pre_module_v
            pre_module_name = k if pre_module_name is None else pre_module_name
            if isinstance(v, nn.BatchNorm2d):

                fusion_conv = fusion.fuse_conv_bn_eval(pre_module_v, v)
                new_modules_list.append(fusion_conv.weight)
                new_modules_list.append(fusion_conv.bias)
                new_modules_name_list.append(pre_module_name+".weight")
                new_modules_name_list.append(pre_module_name+".bias")
            elif not isinstance(pre_module_v, nn.BatchNorm2d) and pre_module_v is not v:
                new_modules_list.append(pre_module_v.weight)
                new_modules_list.append(pre_module_v.bias)
                new_modules_name_list.append(pre_module_name+".weight")
                new_modules_name_list.append(pre_module_name+".bias")
                print(pre_module_name)
            last_module_v = v
            last_module_name = k
            pre_module_v = v
            pre_module_name = k
            
    new_modules_list.append(last_module_v.weight)
    new_modules_list.append(last_module_v.bias)
    new_modules_name_list.append(last_module_name+".weight")
    new_modules_name_list.append(last_module_name+".bias")
    
#    for i, v in enumerate(new_modules_name_list):
#        print(i, v)
    
    # 更新无bn模型参数
    with torch.no_grad():
        for i, (name, param) in enumerate(new_pfld_backbone.named_parameters()):
            param.copy_(new_modules_list[i])
    
    # 误差测试
    _, landmarks = pfld_backbone(input)
    _, new_landmarks = new_pfld_backbone(input)
    print("1-", landmarks)
    print("2-", new_landmarks)
    print(torch.sum((landmarks-new_landmarks)*(landmarks-new_landmarks),axis=1))
    
#    utils.count_params(pfld_backbone, input)
#    utils.count_params(new_pfld_backbone, input)
#    utils.count_interence_time(pfld_backbone, input)
#    utils.count_interence_time(new_pfld_backbone, input)
#    torch.save(new_pfld_backbone.state_dict(),"../pretrained_models/pfld_gray_no_bn.pth.tar")
#    torch.save(pfld_backbone.state_dict(),"../pretrained_models/pfld_gray.pth.tar")
