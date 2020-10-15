import torch
import torch.nn as nn
import collections
def merge_conv_bn(conv, bn):
    print("enter merge_conv_bn >>>")
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma
    merged_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=True)
    merged_conv.weight = nn.Parameter(w)
    merged_conv.bias = nn.Parameter(b)
    print("quit merge_conv_bn <<<")
    return merged_conv
    
def merge(model):
    print("enter merge >>>> ")
    pre_module = None
    cur_module = None
    pre_name = None
    cur_name = None
#    for name, module in model.named_modules():
#        print(name, module)
    model_new = collections.OrderedDict()
    named_modules = list(model.named_modules())
    for name, module in named_modules:
        if(isinstance(module, nn.BatchNorm2d)):
            print("---", name, " is BN")
        if(isinstance(module, nn.Conv2d)):
            print("---", name, " is Conv2d")
        
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Conv2d):
            pre_module = module if pre_module is None else pre_module
            pre_name = name if pre_name is None else pre_name
            if(isinstance(module, nn.BatchNorm2d)):
                print(pre_module)
                print(module)
                merged_conv = merge_conv_bn(pre_module, module)
                model_new[pre_name] = merged_conv
                model_new[name] = nn.Identity()
            pre_module = module
            pre_name = name
        else:
            model_new[name]= module
    
    for name, module in model_new.items():
        print(name)
    
    net = nn.Module
    for n, m in model_new.items():
        print(n, m)
    model.state_dict().update(model_new)
    
    torch.save(model, "model_new.pth")
    
    return model
        
        
