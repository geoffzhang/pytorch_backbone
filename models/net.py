import torch
from torch import nn

# basic block
def conv_bn(inp, oup, kernel=3, stride=1, padding=1, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    features = []
    if norm_layer!=None:
        features.append(conv_layer(inp, oup, kernel, stride, padding, bias=False))
        features.append(norm_layer(oup))
    else:
        features.append(conv_layer(inp, oup, kernel, stride, padding, bias=True))
    features.append(nlin_layer(inplace=True))
    
    return nn.Sequential(*features)

def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    features = []
    if norm_layer!=None:
        features.append(conv_layer(inp, oup, 1, 1, 0, bias=False))
        features.append(norm_layer(oup))
    else:
        features.append(conv_layer(inp, oup, 1, 1, 0, bias=True))
    features.append(nlin_layer(inplace=True))
    return nn.Sequential(*feature)

# depthwise conv and pointwise conv
def conv_dw_pw(inp, oup, kernel, stride, padding, nlin_layer=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel, stride, padding, bias=False, groups=inp),
        nn.BatchNorm2d(inp),
        nlin_layer(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nlin_layer(inplace=True),
    )

#

if __name__=="__main__":
    print("<<<<< enter main <<<<<")
#    conv_bn(112,112, norm_layer=None)
    print("<<<<< quit main <<<<<")
