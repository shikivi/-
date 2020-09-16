import os
import torch as t
import torch.nn as nn

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

    def init(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.checkpoint_dir = opt.checkpoint_dir
        self.device = t.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else t.device('cpu')

    def forward(self, *input):
        return super(BaseNet, self).forward(*input)

    def test(self, *input):
        with t.no_grad():
            self.forward(*input)

    def save_network(self, net, epoch):
        save_name = '%s_net_%s.pth' % (epoch, net)
        save_path = os.path.join(self.checkpoint_dir, save_name)
        t.save(self.cpu().state_dict(), save_path)

    def load_network(self, net, epoch):
        save_name = '%s_net_%s.pth' % (epoch, net)
        save_path = os.path.join(self.checkpoint_dir, save_name)
        self.load_state_dict(t.load(save_path))
