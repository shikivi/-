import torch as t
import torch.nn as nn
import torch.nn.functional as F
from model.basemodel import BaseModel
from model.basenet import BaseNet
from model.loss import WGANLoss, IDMRFLoss
from model.layer import init_weights, PureUpsampling, ConfidenceDrivenMaskLayer, SpectralNorm, GatedConv, GatedDilatedConv
from util.utils import generate_mask
import numpy as np


class GMCNN(BaseNet):
    def __init__(self, in_channels, out_channels, cnum=32, act=F.elu, norm=F.instance_norm, using_norm=False):
        super(GMCNN, self).__init__()
        self.act = act
        self.using_norm = using_norm
        if using_norm is True:
            self.norm = norm
        else:
            self.norm = None
        channel = cnum

        self.EB1 = []
        self.EB2 = []
        self.EB3 = []
        self.decoding_layers = []
        self.EB1_pad = []
        self.EB2_pad = []
        self.EB3_pad = []

        self.EB1.append(GatedConv(in_channels, channel, kernel_size=7, stride=1))
        self.EB1.append(GatedConv(channel, channel * 2, kernel_size=7, stride=2))
        self.EB1.append(GatedConv(channel * 2, channel * 2, kernel_size=7, stride=1))
        self.EB1.append(GatedConv(channel * 2, channel * 4, kernel_size=7, stride=2))
        self.EB1.append(GatedConv(channel * 4, channel * 4, kernel_size=7, stride=1))
        self.EB1.append(GatedConv(channel * 4, channel * 4, kernel_size=7, stride=1))
        self.EB1.append(GatedDilatedConv(channel * 4, channel* 4, kernel_size=7, stride=1, dilation=2))
        self.EB1.append(GatedDilatedConv(channel* 4, channel* 4, kernel_size=7, stride=1, dilation=4))
        self.EB1.append(GatedDilatedConv(channel* 4, channel* 4, kernel_size=7, stride=1, dilation=8))
        self.EB1.append(GatedDilatedConv(channel* 4, channel* 4, kernel_size=7, stride=1, dilation=16))
        self.EB1.append(GatedConv(channel* 4, channel* 4, kernel_size=7, stride=1))
        self.EB1.append(GatedConv(channel* 4, channel* 4, kernel_size=7, stride=1, act=None))
        self.EB1.append(PureUpsampling(scale=4))
        self.EB1_pad = [3, 3, 3, 3, 3, 3, 6, 12, 24, 48, 3, 3, 0]

        self.EB2.append(GatedConv(in_channels, channel, kernel_size=5, stride=1))
        self.EB2.append(GatedConv(channel, channel* 2, kernel_size=5, stride=2))
        self.EB2.append(GatedConv(channel* 2, channel* 2, kernel_size=5, stride=1))
        self.EB2.append(GatedConv(channel* 2, channel* 4, kernel_size=5, stride=2))
        self.EB2.append(GatedConv(channel* 4, channel* 4, kernel_size=5, stride=1))
        self.EB2.append(GatedConv(channel* 4, channel* 4, kernel_size=5, stride=1))
        self.EB2.append(GatedDilatedConv(channel* 4, channel* 4, kernel_size=5, stride=1, dilation=2))
        self.EB2.append(GatedDilatedConv(channel* 4, channel* 4, kernel_size=5, stride=1, dilation=4))
        self.EB2.append(GatedDilatedConv(channel* 4, channel* 4, kernel_size=5, stride=1, dilation=8))
        self.EB2.append(GatedDilatedConv(channel* 4, channel* 4, kernel_size=5, stride=1, dilation=16))
        self.EB2.append(GatedConv(channel* 4, channel* 4, kernel_size=5, stride=1))
        self.EB2.append(GatedConv(channel* 4, channel* 4, kernel_size=5, stride=1,act=None))
        self.EB2.append(PureUpsampling(scale=2, mode='nearest'))
        self.EB2.append(GatedConv(channel* 4, channel* 2, kernel_size=5, stride=1))
        self.EB2.append(GatedConv(channel* 2, channel* 2, kernel_size=5, stride=1,act=None))
        self.EB2.append(PureUpsampling(scale=2))
        self.EB2_pad = [2, 2, 2, 2, 2, 2, 4, 8, 16, 32, 2, 2, 0, 2, 2, 0]

        self.EB3.append(GatedConv(in_channels, channel, kernel_size=3, stride=1))
        self.EB3.append(GatedConv(channel, channel* 2, kernel_size=3, stride=2))
        self.EB3.append(GatedConv(channel* 2, channel* 2, kernel_size=3, stride=1))
        self.EB3.append(GatedConv(channel* 2, channel* 4, kernel_size=3, stride=2))
        self.EB3.append(GatedConv(channel* 4, channel* 4, kernel_size=3, stride=1))
        self.EB3.append(GatedConv(channel* 4, channel* 4, kernel_size=3, stride=1))
        self.EB3.append(GatedDilatedConv(channel* 4, channel* 4, kernel_size=3, stride=1, dilation=2))
        self.EB3.append(GatedDilatedConv(channel* 4, channel* 4, kernel_size=3, stride=1, dilation=4))
        self.EB3.append(GatedDilatedConv(channel* 4, channel* 4, kernel_size=3, stride=1, dilation=8))
        self.EB3.append(GatedDilatedConv(channel* 4, channel* 4, kernel_size=3, stride=1, dilation=16))
        self.EB3.append(GatedConv(channel* 4, channel* 4, kernel_size=3, stride=1))
        self.EB3.append(GatedConv(channel* 4, channel* 4, kernel_size=3, stride=1,act=None))
        self.EB3.append(PureUpsampling(scale=2, mode='nearest'))
        self.EB3.append(GatedConv(channel* 4, channel* 2, kernel_size=3, stride=1))
        self.EB3.append(GatedConv(channel* 2, channel* 2, kernel_size=3, stride=1,act=None))
        self.EB3.append(PureUpsampling(scale=2, mode='nearest'))
        self.EB3.append(GatedConv(channel* 2, channel, kernel_size=3, stride=1))
        self.EB3.append(GatedConv(channel, channel, kernel_size=3, stride=1))
        self.EB3_pad = [1, 1, 1, 1, 1, 1, 2, 4, 8, 16, 1, 1, 0, 1, 1, 0, 1, 1]

        self.decoding_layers.append(GatedConv(channel* 7, channel// 2, kernel_size=3, stride=1,act=None))
        self.decoding_layers.append(GatedConv(channel// 2, out_channels, kernel_size=3, stride=1,act=None))
        self.decoding_pad_rec = [1, 1]

        self.EB1 = nn.ModuleList(self.EB1)
        self.EB2 = nn.ModuleList(self.EB2)
        self.EB3 = nn.ModuleList(self.EB3)
        self.decoding_layers = nn.ModuleList(self.decoding_layers)
        self.pads = [0] * 49
        for i in range(49):
            self.pads[i] = nn.ReflectionPad2d(i)
        self.pads = nn.ModuleList(self.pads)

    def forward(self, x):
        x1, x2, x3 = x, x, x
        for i, layer in enumerate(self.EB1):
            pad_idx = self.EB1_pad[i]
            x1 = layer(self.pads[pad_idx](x1))
            if self.using_norm:
                x1 = self.norm(x1)

        for i, layer in enumerate(self.EB2):
            pad_idx = self.EB2_pad[i]
            x2 = layer(self.pads[pad_idx](x2))
            if self.using_norm:
                x2 = self.norm(x2)

        for i, layer in enumerate(self.EB3):
            pad_idx = self.EB3_pad[i]
            x3 = layer(self.pads[pad_idx](x3))
            if self.using_norm:
                x3 = self.norm(x3)

        x_d = t.cat((x1, x2, x3), 1)
        x_d = self.pads[self.decoding_pad_rec[0]](x_d)
        x_d = self.decoding_layers[0](x_d)
        x_d = self.act(x_d)
        x_d = self.pads[self.decoding_pad_rec[1]](x_d)
        x_d = self.decoding_layers[1](x_d)
        x_out = t.clamp(x_d, -1, 1)
        return x_out



class Discriminator(BaseNet):
    def __init__(self, in_channels, cnum=32, fc_channels=8*8*32*4, act=F.elu, norm=None):
        super(Discriminator, self).__init__()
        self.act = act
        self.norm = norm
        self.embedding = None
        self.logit = None

        channel = cnum
        self.layers = []
        self.layers.append(SpectralNorm(nn.Conv2d(in_channels, channel, kernel_size=5, padding=2, stride=2)))
        self.layers.append(SpectralNorm(nn.Conv2d(channel, channel* 2, kernel_size=5, padding=2, stride=2)))
        self.layers.append(SpectralNorm(nn.Conv2d(channel* 2, channel* 4, kernel_size=5, padding=2, stride=2)))
        self.layers.append(SpectralNorm(nn.Conv2d(channel* 4, channel* 4, kernel_size=5, padding=2, stride=2)))
        self.layers.append(SpectralNorm(nn.Linear(fc_channels, 1)))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        self.embedding = x.view(x.size(0), -1)
        self.logit = self.layers[-1](self.embedding)
        return self.logit



class GlobalLocalDiscriminator(BaseNet):
    def __init__(self, in_channels, cnum=32, g_fc_channels=16*16*32*4, l_fc_channels=8*8*32*4, act=F.elu, norm=None,):
        super(GlobalLocalDiscriminator, self).__init__()
        self.act = act
        self.norm = norm
        self.global_discriminator = Discriminator(in_channels=in_channels, fc_channels=g_fc_channels, cnum=cnum,
                                                  act=act, norm=norm)
        self.local_discriminator = Discriminator(in_channels=in_channels, fc_channels=l_fc_channels, cnum=cnum,
                                                 act=act, norm=norm)

    def forward(self, x_g, x_l):
        return self.global_discriminator(x_g), self.local_discriminator(x_l)



class InpaintingModel_GMCNN(BaseModel):
    def __init__(self, in_channels, act=F.elu, norm=None, opt=None):
        super(InpaintingModel_GMCNN, self).__init__()
        self.opt = opt
        self.init(opt)
        self.confidence_mask_layer = ConfidenceDrivenMaskLayer()
        self.netGM = GMCNN(in_channels, out_channels=3, cnum=opt.g_cnum, act=act, norm=norm).cuda()
        init_weights(self.netGM)
        self.model_names = ['GM']
        if self.opt.phase == 'test':
            return

        self.netD = None
        self.optimizer_G = t.optim.Adam(self.netGM.parameters(), lr=opt.lr, betas=(0.5, 0.9))
        self.optimizer_D = None
        self.wganloss = None
        self.recloss = nn.L1Loss()
        self.aeloss = nn.L1Loss()
        self.mrfloss = None
        self.lambda_adv = opt.lambda_adv
        self.lambda_rec = opt.lambda_rec
        self.lambda_ae = opt.lambda_ae
        self.lambda_gp = opt.lambda_gp
        self.lambda_mrf = opt.lambda_mrf
        self.G_loss, self.G_loss_reconstruction,self. G_loss_mrf = None, None, None
        self.G_loss_adv, self.G_loss_adv_local, self.G_loss_ae = None, None, None
        self.D_loss, self.D_loss_local = None, None
        self.GAN_loss = None
        self.gt, self.gt_local = None, None
        self.mask, self.mask_01 = None, None
        self.rect = None
        self.im_in, self.gin = None, None
        self.completed, self.completed_local = None, None
        self.completed_logit, self.completed_local_logit = None, None
        self.gt_logit, self.gt_local_logit = None, None

        self.pred = None

        if self.opt.pretrain_network is False:
            img_channels=opt.img_shapes[0]//16*opt.img_shapes[1]//16*opt.d_cnum*4
            mask_channels= l_fc_channels=opt.mask_shapes[0]//16*opt.mask_shapes[1]//16*opt.d_cnum*4
            if self.opt.mask_type == 'rect':
                self.netD = GlobalLocalDiscriminator(3, cnum=opt.d_cnum, act=act, g_fc_channels=img_channels,
                                                     l_fc_channels=mask_channels,).cuda()
            else:
                self.netD = GlobalLocalDiscriminator(3, cnum=opt.d_cnum, act=act, g_fc_channels=img_channels,
                                                     l_fc_channels=img_channels).cuda()
            init_weights(self.netD)
            self.optimizer_D = t.optim.Adam(filter(lambda x: x.requires_grad, self.netD.parameters()),
                                            lr=opt.lr, betas=(0.5, 0.9))
            self.wganloss = WGANLoss()
            self.mrfloss = IDMRFLoss()

    def initVariables(self):
        self.gt = self.input['gt']
        mask, rect = generate_mask(self.opt.mask_type, self.opt.img_shapes, self.opt.mask_shapes)
        self.mask_01 = t.from_numpy(mask).cuda().repeat([self.opt.batch_size, 1, 1, 1])
        self.mask = self.confidence_mask_layer(self.mask_01)
        if self.opt.mask_type == 'rect':
            self.rect = [rect[0, 0], rect[0, 1], rect[0, 2], rect[0, 3]]
            self.gt_local = self.gt[:, :, self.rect[0]:self.rect[0] + self.rect[1],
                            self.rect[2]:self.rect[2] + self.rect[3]]
        else:
            self.gt_local = self.gt
        self.im_in = self.gt * (1 - self.mask_01)
        self.gin = t.cat((self.im_in, self.mask_01), 1)

    #loss部分有点不理解实际意义
    def forward_G(self):
        self.G_loss_reconstruction = self.recloss(self.completed * self.mask, self.gt.detach() * self.mask)
        self.G_loss_reconstruction = self.G_loss_reconstruction / t.mean(self.mask_01)
        self.G_loss_ae = self.aeloss(self.pred * (1 - self.mask_01), self.gt.detach() * (1 - self.mask_01))
        self.G_loss_ae = self.G_loss_ae / t.mean(1 - self.mask_01)
        self.G_loss = self.lambda_rec * self.G_loss_reconstruction + self.lambda_ae * self.G_loss_ae

        if self.opt.pretrain_network is False:
            self.completed_logit, self.completed_local_logit = self.netD(self.completed, self.completed_local)
            self.G_loss_mrf = self.mrfloss((self.completed_local+1)/2.0, (self.gt_local.detach()+1)/2.0)
            self.G_loss = self.G_loss + self.lambda_mrf * self.G_loss_mrf
            self.G_loss_adv = -self.completed_logit.mean()
            self.G_loss_adv_local = -self.completed_local_logit.mean()
            self.G_loss = self.G_loss + self.lambda_adv * (self.G_loss_adv + self.G_loss_adv_local)

    def forward_D(self):
        self.completed_logit, self.completed_local_logit = self.netD(self.completed.detach(), self.completed_local.detach())
        self.gt_logit, self.gt_local_logit = self.netD(self.gt, self.gt_local)
        self.D_loss_local = nn.ReLU()(1.0 - self.gt_local_logit).mean() + nn.ReLU()(1.0 + self.completed_local_logit).mean()
        self.D_loss = nn.ReLU()(1.0 - self.gt_logit).mean() + nn.ReLU()(1.0 + self.completed_logit).mean()
        self.D_loss = self.D_loss + self.D_loss_local

    def backward_G(self):
        self.G_loss.backward()

    def backward_D(self):
        self.D_loss.backward(retain_graph=True)

    def optimize_parameters(self):
        self.initVariables()

        self.pred = self.netGM(self.gin)
        self.completed = self.pred * self.mask_01 + self.gt * (1 - self.mask_01)
        if self.opt.mask_type == 'rect':
            self.completed_local = self.completed[:, :, self.rect[0]:self.rect[0] + self.rect[1],
                                   self.rect[2]:self.rect[2] + self.rect[3]]
        else:
            self.completed_local = self.completed

        if self.opt.pretrain_network is False:
            for i in range(self.opt.D_max_iters):
                self.optimizer_D.zero_grad()
                self.optimizer_G.zero_grad()
                self.forward_D()
                self.backward_D()
                self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.forward_G()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_losses(self):
        l = {'G_loss': self.G_loss.item(), 'G_loss_rec': self.G_loss_reconstruction.item(),
             'G_loss_ae': self.G_loss_ae.item()}
        if self.opt.pretrain_network is False:
            l.update({'G_loss_adv': self.G_loss_adv.item(),
                      'G_loss_adv_local': self.G_loss_adv_local.item(),
                      'D_loss': self.D_loss.item(),
                      'G_loss_mrf': self.G_loss_mrf.item()})
        return l


    def get_current_visuals_tensor(self):
        return {'input': self.im_in.cpu().detach(), 'gt': self.gt.cpu().detach(),
                'completed': self.completed.cpu().detach()}

    def evaluate(self, img, mask):
        img = t.from_numpy(img).type(t.FloatTensor).cuda() / 127.5 - 1
        mask = t.from_numpy(mask).type(t.FloatTensor).cuda()
        img = img * (1-mask)
        xin = t.cat((img, mask), 1)
        ret = (self.netGM(xin) * mask + img * (1-mask)).cpu().detach().numpy()
        ret = (ret+ 1) * 127.5
        uint8_rect=ret.astype(np.uint8)
        return uint8_rect
