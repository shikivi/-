import torch as t
import torch.nn as nn
import torch.nn.functional as F
from util.utils import gauss_kernel
import torchvision.models as models



class PureUpsampling(nn.Module):
    def __init__(self, scale=2, mode='bilinear'):
        super(PureUpsampling, self).__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, input):
        height= input.size(2) * self.scale
        width=input.size(3) * self.scale
        if self.mode == 'nearest':
            out = F.interpolate(input, (height, width), mode=self.mode)
        else:
            out = F.interpolate(input, (height, width), mode=self.mode, align_corners=True)
        return out


class GaussianBlurLayer(nn.Module):
    def __init__(self, size, sigma, in_channel=1, stride=1, padding=1):
        super(GaussianBlurLayer, self).__init__()
        self.size = size
        self.sigma = sigma
        self.channel = in_channel
        self.stride = stride
        self.padding = nn.ReflectionPad2d(padding)

    def forward(self, input):
        kernel = gauss_kernel(self.size, self.sigma, self.channel, self.channel)
        kernel = t.from_numpy(kernel).cuda()
        blurred = F.conv2d(self.padding(input), kernel, stride=self.stride)
        return blurred


class ConfidenceDrivenMaskLayer(nn.Module):
    def __init__(self, size=65, sigma=1.0/40, iters=7):
        super(ConfidenceDrivenMaskLayer, self).__init__()
        self.size = size
        self.sigma = sigma
        self.iters = iters
        self.propagationLayer = GaussianBlurLayer(size, sigma, padding=32)

    def forward(self, mask):
        init = 1 - mask
        mask_confidence = None
        for i in range(self.iters):
            mask_confidence = self.propagationLayer(init)* mask
            init = mask_confidence + (1 - mask)
        return mask_confidence


class VGG19(nn.Module):
    def __init__(self, pool='max'):
        super(VGG19, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return out


class VGG19featureLayer(nn.Module):
    def __init__(self):
        super(VGG19featureLayer, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True).features.eval().cuda()
        self.mean = t.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()

    def forward(self, input):
        out = {}
        input = input - self.mean
        part = 1
        no = 0
        for layer in self.vgg19.children():
            if isinstance(layer, nn.Conv2d):
                no += 1
                name = 'conv{}_{}'.format(part, no)
            elif isinstance(layer, nn.ReLU):
                no += 1
                name = 'relu{}_{}'.format(part, no)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                no = 0
                name = 'pool_{}'.format(no)
                part += 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(part)
            input = layer(input)
            out[name] = input
        return out


def init_weights(net, gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            nn.init.normal_(m.weight.data, 0.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def myNormalize(data, eps=1e-12):
    return data / (data.norm()+eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iteration=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iteration = power_iteration
        if self._made_params():
            pass
        else:
            self._make_params()

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + '_u')
            v = getattr(self.module, self.name + '_v')
            w = getattr(self.module, self.name + '_bar')
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = myNormalize(u.data)
        v.data = myNormalize(v.data)
        w_bar = nn.Parameter(w.data)
        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + '_u', u)
        self.module.register_parameter(self.name + '_v', v)
        self.module.register_parameter(self.name + '_bar', w_bar)

    def _update_u_v(self):
        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')
        w = getattr(self.module, self.name + '_bar')
        height = w.data.shape[0]

        for i in range(self.power_iteration):
            w_data=w.view(height, -1).data
            v.data = myNormalize(t.mv(t.t(w_data), u.data))
            u.data = myNormalize(t.mv(w_data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def forward(self, *input):
        self._update_u_v()
        return self.module.forward(*input)


class GatedConv(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, kernel_size=3, stride=1, act=F.elu):
        super(GatedConv, self).__init__()
        self.ksize = kernel_size
        self.stride = stride
        self.act = act
        self.convf = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=kernel_size)
        self.convm = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=kernel_size)

    def forward(self, in_x):
        x = self.convf(in_x)
        if self.act is not None:
            x = self.act(x)
        m = self.convm(in_x)
        m = t.sigmoid(m)
        return x * m


class GatedDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=2, act=F.elu):
        super(GatedDilatedConv, self).__init__()
        self.ksize = kernel_size
        self.stride = stride
        self.act = act
        self.convf = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=kernel_size, dilation=dilation)
        self.convm = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=kernel_size, dilation=dilation)

    def forward(self, in_x):
        x = self.convf(in_x)
        x = self.act(x)
        m = self.convm(in_x)
        m = t.sigmoid(m)
        return x * m
