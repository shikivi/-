import torch as t
import torch.nn as nn
import torch.nn.functional as F
from model.layer import VGG19featureLayer
from functools import reduce

class WGANLoss(nn.Module):
    def __init__(self):
        super(WGANLoss, self).__init__()

    def __call__(self, input, target):
        d_loss = (input - target).mean()
        g_loss = -(input.mean())
        return {'g_loss': g_loss, 'd_loss': d_loss}


class IDMRFLoss(nn.Module):
    def __init__(self, featureLayer=VGG19featureLayer):
        super(IDMRFLoss, self).__init__()
        self.featureLayer = featureLayer()
        self.style_layers = {'relu3_2': 1.0, 'relu4_2': 1.0}
        self.content_layers = {'relu4_2': 1.0}
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.l_s = 1.0
        self.l_c = 1.0

    def patch_extraction(self, featureMaps):
        size = 1
        stride = 1
        patches_as_depth_vectors = featureMaps.unfold(2, size, stride).unfold(3, size, stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def exp_norm_relative_dist(self, relative):
        dist = t.exp((self.bias - relative)/self.nn_stretch_sigma)
        self.cs_NCHW = dist / t.sum(dist, dim=1, keepdim=True)
        return self.cs_NCHW

    def mrf_loss(self, gen, target):
        meanTemp = t.mean(target, 1, keepdim=True)
        gen_feats = gen - meanTemp
        gen_norm = t.norm(gen_feats, p=2, dim=1, keepdim=True)
        target_feats= target - meanTemp
        target_norm = t.norm(target_feats, p=2, dim=1, keepdim=True)
        gen_normalized = gen_feats / gen_norm
        target_normalized = target_feats / target_norm

        cosine_dist_l = []
        BatchSize = target.size(0)
        for i in range(BatchSize):
            target_feat_i = target_normalized[i:i+1, :, :, :]
            gen_feat_i = gen_normalized[i:i+1, :, :, :]
            patches_OIHW = self.patch_extraction(target_feat_i)
            cosine_dist_l.append(F.conv2d(gen_feat_i, patches_OIHW))

        cosine_dist = - (t.cat(cosine_dist_l, dim=0) - 1) / 2
        relative_dist = (cosine_dist) / (t.min(cosine_dist, dim=1, keepdim=True)[0] + 1e-5)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = t.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = t.mean(k_max_nc, dim=1)
        div_mrf_sum = t.sum(-t.log(div_mrf))
        return div_mrf_sum

    def forward(self, gen, target):
        gen_feats = self.featureLayer(gen)
        tar_feats = self.featureLayer(target)
        style_loss_list=[]
        for layer in self.style_layers:
            style_loss_list.append(self.style_layers[layer] * self.mrf_loss(gen_feats[layer], tar_feats[layer]))
        self.style_loss = reduce(lambda x, y: x+y, style_loss_list) * self.l_s
        content_loss_list=[]
        for layer in self.content_layers:
            content_loss_list.append(self.content_layers[layer] * self.mrf_loss(gen_feats[layer], tar_feats[layer]))
        self.content_loss = reduce(lambda x, y: x+y, content_loss_list) * self.l_c
        return self.style_loss + self.content_loss

