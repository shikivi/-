import numpy as np
import cv2
import os
from options.test_options import TestOptions
from model.net import InpaintingModel_GMCNN
from util.utils import generate_my_mask, getLatest


config = TestOptions().parse()

ourModel = InpaintingModel_GMCNN(in_channels=4, opt=config)

if config.load_model_dir != '':
    ourModel.load_networks(getLatest(os.path.join(config.load_model_dir, '*.pth')))


def generate(pathfile,x1,y1,x2,y2,i):
    mask, _ = generate_my_mask(config.img_shapes, x1,y1,x2-x1,y2-y1)
    image = cv2.imread(pathfile)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)

    h, w = image.shape[2:]
    grid = 4
    image = image[:, :, :h // grid * grid, :w // grid * grid]
    mask = mask[:, :, :h // grid * grid, :w // grid * grid]
    result = ourModel.evaluate(image, mask)
    result = np.transpose(result[0][::-1,:,:], [1, 2, 0])
    cv2.imwrite(os.path.join(config.saving_path, '{:03d}.png'.format(i)), result)


def generate_pic(rect_path,img_path):
    rect_name = rect_path
    x=[]
    with open(rect_name, 'r+', encoding='utf-8') as f:
        for line in f.readlines():
            x.append((list(map(float,line[:-1].split(',')))))

    img_name = img_path
    name=[]
    with open(img_name, 'r+', encoding='utf-8') as f:
        for line in f.readlines():
            name.append(line[:-1])
    for i in range(291):
        generate(name[i],int(x[i][1]-2),int(x[i][0]-2),int(x[i][1]+x[i][3]+3),int(x[i][0]+x[i][2]+3),i+1)




