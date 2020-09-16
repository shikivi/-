import os
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from data.data import InpaintingDataset, ToTensor
from model.net import InpaintingModel_GMCNN
from options.train_options import TrainOptions
from util.utils import getLatest

config = TrainOptions().parse()

print('loading data..')
dataset = InpaintingDataset(config.dataset_path, '', transform=transforms.Compose( [ToTensor()] ))
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=True)
print('data loaded..')

print('Preparing model..')
ourModel = InpaintingModel_GMCNN(in_channels=4, opt=config)

if config.load_model_dir != '':
    ourModel.load_networks(getLatest(os.path.join(config.load_model_dir, '*.pth')))

print('Initializing training..')

for epoch in range(config.epochs):

    for i, data in enumerate(dataloader):
        gt = data['gt'].cuda()
        gt = gt / 127.5 - 1

        data_in = {'gt': gt}
        ourModel.setInput(data_in)
        ourModel.optimize_parameters()

        if (i+1) % config.viz_steps == 0:
            loss = ourModel.get_current_losses()
            if config.pretrain_network is False:
                print(
                    '[%d, %5d] G_loss: %.4f (rec: %.4f, ae: %.4f, adv: %.4f, mrf: %.4f), D_loss: %.4f'
                    % (epoch + 1, i + 1, loss['G_loss'], loss['G_loss_rec'], loss['G_loss_ae'],
                       loss['G_loss_adv'], loss['G_loss_mrf'], loss['D_loss']))
            else:
                print('[%d, %5d] G_loss: %.4f (rec: %.4f, ae: %.4f)'
                      % (epoch + 1, i + 1, loss['G_loss'], loss['G_loss_rec'], loss['G_loss_ae']))

            images = ourModel.get_current_visuals_tensor()
            image_completed = vutils.make_grid(images['completed'], normalize=True, scale_each=True)
            image_input = vutils.make_grid(images['input'], normalize=True, scale_each=True)
            image_gt = vutils.make_grid(images['gt'], normalize=True, scale_each=True)
            if (i+1) % config.train_spe == 0:
                print('saving model ..')
                ourModel.save_networks(epoch+1)
    ourModel.save_networks(epoch+1)

