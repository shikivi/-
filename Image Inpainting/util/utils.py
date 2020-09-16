import numpy as np
import scipy.stats as st
import cv2
import time
import os
import glob

def gauss_kernel(size=21, sigma=3, channel_input=3, channel_output=3):
    interval = (2 * sigma + 1.0) / size
    x = np.linspace(-sigma-interval/2,sigma+interval/2,size+1)
    tmp_ker = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(tmp_ker, tmp_ker))
    kernel = kernel_raw / kernel_raw.sum()
    filter = np.array(kernel, dtype=np.float32).reshape((1, 1, size, size))
    out_filter = np.tile(filter, [channel_output, channel_input, 1, 1])
    return out_filter


def np_free_form_mask(maxVertex, maxLen, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        len = np.random.randint(maxLen + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + len * np.cos(angle)
        nextX = startX + len * np.sin(angle)

        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)

        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)

        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask


def generate_rect_mask(image_size, mask_size, margin=8, random_mask=True):
    mask = np.zeros((image_size[0], image_size[1])).astype(np.float32)
    if random_mask:
        size0, size1 = mask_size[0], mask_size[1]
        loc0 = np.random.randint(margin, image_size[0] - size0 - margin)
        loc1 = np.random.randint(margin, image_size[1] - size1 - margin)
    else:
        size0, size1 = mask_size[0], mask_size[1]
        loc0 = (image_size[0] - size0) // 2
        loc1 = (image_size[1] - size1) // 2
    mask[loc0:loc0+size0, loc1:loc1+size1] = 1
    mask = np.expand_dims(mask, axis=0)
    mask = np.expand_dims(mask, axis=0)
    rect = np.array([[loc0, size0, loc1, size1]], dtype=int)
    return mask, rect

def generate_mask(image_size, loc0, loc1, size0, size1):
    mask = np.zeros((image_size[0], image_size[1])).astype(np.float32)
    mask[loc0:loc0+size0, loc1:loc1+size1] = 1
    mask = np.expand_dims(mask, axis=0)
    mask = np.expand_dims(mask, axis=0)
    rect = np.array([[loc0, size0, loc1, size1]], dtype=int)
    return mask, rect

def generate_stroke_mask(image_size, parts=10, maxVertex=20, maxLen=100, maxBrushWidth=24, maxAngle=360):
    mask = np.zeros((image_size[0], image_size[1], 1), dtype=np.float32)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLen, maxBrushWidth, maxAngle, image_size[0], image_size[1])
    mask = np.transpose(np.minimum(mask, 1.0), [2, 0, 1])
    mask = np.expand_dims(mask, 0)
    return mask

def generate_mask(type, image_size, mask_size):
    if type == 'rect':
        return generate_rect_mask(image_size, mask_size)
    else:
        return generate_stroke_mask(image_size), None


def getLatest(folder_path):
    files = glob.glob(folder_path)
    file_times = list(map(lambda x: time.ctime(os.path.getctime(x)), files))
    return files[sorted(range(len(file_times)), key=lambda x: file_times[x])[-1]]
