# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import argparse
from models.pfld_no_bn import PFLDInference
from torch.autograd import Variable
import torch
import onnxsim
import numpy as np
import onnxruntime

parser = argparse.ArgumentParser(description='pytorch2onnx')
parser.add_argument('--torch_model', default="./pretrained_models/pfld_gray_no_bn.pth.tar")
parser.add_argument('--onnx_model', default="./output/pfld_gray_no_bn.onnx")
parser.add_argument('--onnx_model_sim', help='Output ONNX model', default="./output/pfld_gray_no_bn_sim.onnx")
args = parser.parse_args()

print("=====> load pytorch checkpoint...")
checkpoint = torch.load(args.torch_model, map_location=torch.device('cpu'))
plfd_backbone = PFLDInference(136)
plfd_backbone.load_state_dict(checkpoint)
#print("PFLD bachbone:", plfd_backbone)

print("=====> convert pytorch model to onnx...")
dummy_input = Variable(torch.randn(1, 1, 112, 112)) 
input_names = ["input_1"]
output_names = [ "output_1" ]
torch.onnx.export(plfd_backbone, dummy_input, args.onnx_model, opset_version=11, verbose=True, input_names=input_names, output_names=output_names)


print("====> check onnx model...")
import onnx
model = onnx.load(args.onnx_model)
onnx.checker.check_model(model)

print("====> Simplifying...")
model_opt, check = onnxsim.simplify(model)
assert check, "Simplified ONNX model could not be validated"

print("====> Save ...")
onnx.save(model_opt, args.onnx_model_sim)
print("onnx model simplify Ok!")

# verfied model
# pytorch
plfd_backbone.eval()
input_pytorch = torch.randn(1, 1, 112, 112)*0+1
_, landmarks = plfd_backbone(input_pytorch)
print("pytorch", landmarks)

#onnx
input_onnx = np.random.rand(1,1,112,112).astype(np.float32) * 0+1
session = onnxruntime.InferenceSession(args.onnx_model_sim, None)
input_name = session.get_inputs()[0].name
output = session.run([], {input_name: input_onnx})[1]
print("onnx", output)







