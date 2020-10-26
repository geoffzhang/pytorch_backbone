import torch.nn as nn
import torch.functional as F
import torch

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# basic block
def conv_bn(inp, oup, kernel=3, stride=1, padding=1, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    features = []
    if norm_layer!=None:
        features.append(conv_layer(inp, oup, kernel, stride, padding, bias=False))
        features.append(norm_layer(oup))
    else:
        features.append(conv_layer(inp, oup, kernel, stride, padding, bias=True))
    
    if nlin_layer!=None:
        features.append(nlin_layer(inplace=True))
    
    return nn.Sequential(*features)

def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    features = []
    if norm_layer!=None:
        features.append(conv_layer(inp, oup, 1, 1, 0, bias=False))
        features.append(norm_layer(oup))
    else:
        features.append(conv_layer(inp, oup, 1, 1, 0, bias=True))
    
    if nlin_layer!=None:
        features.append(nlin_layer(inplace=True))
    return nn.Sequential(*features)

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

# SE Module
class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            # Hsigmoid()
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# MS-CAM
class MSCAM(nn.Module):
    def __init__(self, inp, oup, r):
        super(MSCAM, self).__init__()
        branch1 = []
        branch1.append(nn.AdaptiveAvgPool2d(1))
        oup_medium = _make_divisible(inp*r, 4)
        branch1.append(conv_1x1_bn(inp, oup_medium, nn.Conv2d, nn.BatchNorm2d, nn.ReLU))
        branch1.append(conv_1x1_bn(oup_medium, inp, nn.Conv2d, nn.BatchNorm2d, None))
        self.branch1 = nn.Sequential(*branch1)

        branch2 = []
        branch2.append(conv_1x1_bn(inp, oup_medium, nn.Conv2d, nn.BatchNorm2d, nn.ReLU))
        branch2.append(conv_1x1_bn(oup_medium, inp, nn.Conv2d, nn.BatchNorm2d, None))
        self.branch2 = nn.Sequential(*branch2)
    
    def forward(self, x):
#        x1 = self.branch1(x)
        x2 = self.branch2(x)

#        x = x * torch.sigmoid(x1+x2)  # sigmoid(x1+x2)

        return x

# AFF
class AFF(nn.Module):
    def __init__(self, inp, oup, r):
        super(AFF, self).__init__()
        self.mscam = MSCAM(inp, oup, r)

    def forward(self, x, y):
        mscam = self.mscam(x+y)

        return x*mscam+y*(1-mscam)

if __name__=="__main__":
    import sys
    sys.path.append("/home/geoff/workspace/github_mine/pytorch_backbone")
    import utils.utils_base as utils

    print("<<<<< enter main <<<<<")
#    conv_bn(112,112, norm_layer=None)
    input = torch.randn(1, 32, 224, 224)
    input1 = torch.randn(1, 32, 224, 224)

    mscam = MSCAM(32,32,0.25)
    mscam(input)

    se = SEModule(32)
    utils.count_interence_time(mscam, input)

    # aff = AFF(32,32,0.5)
    # aff(input, input1)
    # print(aff)
    print("<<<<< quit main <<<<<")
