import torch
import time

from tensorboardX import SummaryWriter
from torchviz import make_dot
import tensorwatch as tw

# 统计模型的参数量
def count_params(net, input):

    net_dict = net.state_dict()
    params = list(net.parameters())
    
#    all_layers_params_sum = 0
#    for i, layer in enumerate(params):
#        single_layer_params_sum = 1
#        for j in layer.size():
#            single_layer_params_sum *= j
##        print("{}: {} {}".format(i, list(layer.size()), single_layer_params_sum))
#        all_layers_params_sum = all_layers_params_sum + single_layer_params_sum
#    print("Total: {:.2f}M".format(all_layers_params_sum/1000000.0))
    
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))

# 统计模型的接口时间
def count_interence_time(net, input):
    count = 0
    total_time = 0
    while True:
        start = time.time()
        output = net(input)
        end = time.time()
        once_time = (end - start)*1000
        total_time = total_time + once_time
        count = count + 1
        print("count: {}, once time: {:.2f}ms, avr time: {:.2f}ms".format(count, once_time, total_time/count))

# 保存可视化模型
def draw_visual_net(net, input , filename="", method="torchviz"):
    
    if method == "torchviz":
        output = net(input)
        g = make_dot(output)
#        g = make_dot(output, params=dict(backbone.named_parameters()))
#        g = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
#        g.view() # 会生成一个 Digraph.gv.pdf 的PDF文件
        g.render(filename+'_net', view=False) # 会自动保存为一个 espnet.pdf，第二个参数为True,则会自动打开该PDF文件，为False则不打开
    
    elif method == "tensorboard":
        with SummaryWriter(comment='mnt') as w:
            w.add_graph(net, input)

    elif method == "tensorwatch":
        img = tw.draw_model(net, input)
        img.save(r'./alexnet.jpg')

