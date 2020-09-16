import os
import torch as t
import torch.nn as nn

# 先整一个复合的大类，只定义框架
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel,self).__init__()

    def init(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.save_dir = opt.model_folder
        self.device = t.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else t.device('cpu')
        self.model_names = []

    def setInput(self, input):
        self.input = input

    def forward(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def update_learning_rate(self):
        pass

    def test(self):
        with t.no_grad():
            self.forward()

    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_name = os.path.join(self.save_dir, '%s_net_%s.pth' % (epoch, name))
                net = getattr(self, 'net' + name)
                t.save(net.state_dict(), save_name)


    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)


    def load_networks(self, load_path):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                if isinstance(net, t.nn.DataParallel):
                    net = net.module
                data = t.load(load_path)
                for key in list(data.keys()):
                    self.__patch_instance_norm_state_dict(data, net, key.split('.'))
                net.load_state_dict(data)


    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
