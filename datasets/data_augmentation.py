from PIL import Image, ImageEnhance
import numpy as np
from args import ARGS

def data_augmentation(img, label, edge):
    r_img, r_label, r_edge = img, label, edge
    r_img, r_label, r_edge = _random_mirror(r_img, r_label, r_edge)
    r_img, r_label, r_edge = _random_scale(r_img, r_label, r_edge)
    r_img, r_label, r_edge = _random_rotation(r_img, r_label, r_edge)
    r_img = _random_color_jitter(r_img)
    return r_img

def _random_mirror(img, label, edge):
    r_img, r_label, r_edge = img, label, edge
    if np.random.random() < 0.5:
        r_img = r_img.transpose(Image.FLIP_LEFT_RIGHT)
        r_label = r_label.transpose(Image.FLIP_LEFT_RIGHT)
        r_edge = r_edge.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.random() < 0.5:
        r_img = r_img.transpose(Image.FLIP_TOP_BOTTOM)
        r_label = r_label.transpose(Image.FLIP_TOP_BOTTOM)
        r_edge = r_edge.transpose(Image.FLIP_TOP_BOTTOM)
    return r_img, r_label, r_edge

def _random_scale(img, label, edge):
    r_img, r_label, r_edge = img, label, edge
    if np.random.random() < 0.05:
        z = np.random.uniform(0.8, 1.2) # 0.5 ~ 2
        width, height = img.size
        to_width, to_height = int(z*width), int(z*height)
        r_img = img.resize((to_width, to_height), Image.ANTIALIAS)
        r_label = label.resize((to_width, to_height), Image.ANTIALIAS)
        r_edge = edge.resize((to_width, to_height), Image.ANTIALIAS)
    return r_img, r_label, r_edge

def _random_rotation(img, label, edge):
    r_img, r_label, r_edge = img, label, edge
    if np.random.random() < 0.5:
        theta = np.random.uniform(-10, 10)
        r_img = img.rotate(theta)
        r_label = label.rotate(theta)
        r_edge = edge.rotate(theta)
    return r_img, r_label, r_edge

def _random_color_jitter(img):
    r_img = img
    transform_tuples = [
        (ImageEnhance.Brightness, 0.1026),
        (ImageEnhance.Contrast, 0.0935),
        (ImageEnhance.Sharpness, 0.8386),
        (ImageEnhance.Color, 0.1592)
    ]
    if np.random.random() < 0.5:
        rand_num = np.random.uniform(0, 1, len(transform_tuples))
        for i, (transformer, alpha) in enumerate(transform_tuples):
            r = alpha * (rand_num[i]*2.0 - 1.0) + 1   # r in [1-alpha, 1+alpha)
            r_img = transformer(r_img).enhance(r)
    return r_img

def _random_crop(img, label, edge):
    r_img, r_label, r_edge = img, label, edge
    width, height = img.size
    r_width, r_height = ARGS['crop_size'], ARGS['crop_size'] # 512, 512
    zx, zy = np.random.randint(0, width - r_width), np.random.randint(0, height - r_height)
    r_img = r_img.crop((zx, zy, zx+r_width, zy+r_height))
    r_label = r_label.crop((zx, zy, zx+r_width, zy+r_height))
    r_edge = r_edge.crop((zx, zy, zx+r_width, zy+r_height))
    return r_img, r_label, r_edge