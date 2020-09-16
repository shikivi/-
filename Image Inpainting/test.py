import numpy as np
import cv2
import os
import subprocess
import glob
from options.test_options import TestOptions
from model.net import InpaintingModel_GMCNN
from util.utils import generate_rect_mask, generate_stroke_mask, getLatest


config = TestOptions().parse()

if os.path.isfile(config.dataset_path):
    pathfile = open(config.dataset_path, 'rt').read().splitlines()
elif os.path.isdir(config.dataset_path):
    pathfile = glob.glob(os.path.join(config.dataset_path, '*.png'))
else:
    print('Fail to get Test set.')
    exit(1)
total_number = len(pathfile)
test_num = total_number if config.test_num == -1 else min(total_number, config.test_num)
print('Totally {} images in test-set, using {} images from it.'.format(total_number, test_num))

print('Preparing model..')
ourModel = InpaintingModel_GMCNN(in_channels=4, opt=config)
ourModel.print_networks()
if config.load_model_dir != '':
    print('Loading pretrained model from {}'.format(config.load_model_dir))
    ourModel.load_networks(getLatest(os.path.join(config.load_model_dir, '*.pth')))
    print('Load done.')

if config.random_mask:
    np.random.seed(config.seed)

for i in range(test_num):
    if config.mask_type == 'rect':
        mask, _ = generate_rect_mask(config.img_shapes, config.mask_shapes, config.random_mask)
    else:
        mask = generate_stroke_mask(image_size=(config.img_shapes[0], config.img_shapes[1]),
                                    parts=8, maxBrushWidth=20, maxLen=100, maxVertex=20)
    image = cv2.cvtColor(cv2.imread(pathfile[i]), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    if h >= config.img_shapes[0] and w >= config.img_shapes[1]:
        h_start = (h-config.img_shapes[0]) // 2
        w_start = (w-config.img_shapes[1]) // 2
        image = image[h_start: h_start+config.img_shapes[0], w_start: w_start+config.img_shapes[1], :]
    else:
        tmp = min(h, w)
        image = image[(h-tmp)//2:(h-tmp)//2+tmp, (w-tmp)//2:(w-tmp)//2+tmp, :]
        image = cv2.resize(image, (config.img_shapes[1], config.img_shapes[0]))

    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image_masked = image * (1-mask) + 255 * mask
    image_masked = np.transpose(image_masked[0][::-1,:,:], [1, 2, 0])
    cv2.imwrite(os.path.join(config.saving_path, '{:03d}_input.png'.format(i)), image_masked.astype(np.uint8))

    h, w = image.shape[2:]
    grid = 4
    image = image[:, :, :h // grid * grid, :w // grid * grid]
    mask = mask[:, :, :h // grid * grid, :w // grid * grid]
    result = ourModel.evaluate(image, mask)
    result = np.transpose(result[0][::-1,:,:], [1, 2, 0])
    cv2.imwrite(os.path.join(config.saving_path, '{:03d}.png'.format(i)), result)
    print(' > {} / {}'.format(i+1, test_num))
print('done.')
