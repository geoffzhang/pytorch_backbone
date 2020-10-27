import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )
def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )
            
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

class MobileNetV1(nn.Module):
    def __init__(self, n_class=1000, 
                       width_mult = 1.0,
                       round_nearest=8):
        super(MobileNetV1, self).__init__()
        self.nclass = n_class

        input_channel = 32
        last_channel = 1024

        dw_setting = [
        #    inp   oup    s
            [32,   64,    1],
            [64,   128,   2],
            [128,  128,   1],
            [128,  256,   2],
            [256,  256,   1],
            [256,  512,   2],
            [512,  512,   1],
            [512,  512,   1],
            [512,  512,   1],
            [512,  512,   1],
            [512,  512,   1],
            [512,  1024,  2],
        ]
        
        # build first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [conv_bn(3, input_channel, stride=2)]
        
        # building dw
        for inp, out, s in dw_setting:
            output_channel = _make_divisible(out * width_mult, round_nearest)
            stride = s
            features.append(conv_dw(input_channel, output_channel, stride))
            input_channel = output_channel

        features.append(conv_dw(input_channel, self.last_channel, stride=1))
        
        self.features = nn.Sequential(*features)

#        self.classifier = nn.Linear(self.last_channel, self.nclass)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
#        x = x.view(-1, self.last_channel)
#        x = self.classifier(x)
        return x

if __name__ == '__main__':
    import sys
    sys.path.append("/home/geoff/workspace/github/framework/pytorch_backbone")
    from utils import utils_base as utils
    
    device = torch.device("cpu") 
    input = torch.randn(1, 3, 640, 480)
    net = MobileNetV1(width_mult=0.25).to(device)
    net.eval()
    
    # utils.count_interence_time(net, input)
    utils.count_params(net, input)

    torch.save(net.state_dict(), "v1_0.25.pth")

