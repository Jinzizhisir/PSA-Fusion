import math
import numpy as np
from PIL import Image,ImageFilter
from Rotation import Rotation
import tensorflow as tf
# import tensorflow_addons as tfa
import tensorflow_addons as tfa


def Rolling_Shutter(input_image,rolling_speed=10800.0,scanning_speed=15000.0,exposure_rate=108.0):
    def add_margin(img, top, right, bottom, left, color):
        top = int(top)
        right = int(right)
        bottom = int(bottom)
        left = int(left)
        paddings = tf.constant([[0,0],[left,right],[top,bottom],[0,0]],dtype=tf.int32)
        result = tf.pad(img,paddings,'CONSTANT',constant_values=255)
        return result

    def add_exposure(img, angle, exposure, row, n):
        new_img = tf.constant(0, shape=(img.shape[1], img.shape[2], img.shape[3]),dtype=tf.int32)
        for i in range(n + 1):
            figs = tfa.image.rotate(img,angles=angle + i/n*exposure)
            figs = tf.cast(figs,dtype=tf.int32)
            new_img = tf.math.add(new_img,figs)
        new_img = tf.math.divide(new_img,n+1)
        return new_img

    im = input_image
    # im = im.filter(ImageFilter.GaussianBlur(radius=0.1))
    # im = im.filter(ImageFilter.SMOOTH_MORE)
    S1 = im.shape[1]
    S2 = im.shape[2]
    S3 = im.shape[3]
    # R = int(math.ceil(math.sqrt(S1 * S1 + S2 * S2)))
    R = max(S1, S2)

    if (R - S1) % 2 == 0 and (R - S2) % 2 == 0:
        im = add_margin(im, (R - S1) / 2, (R - S2) / 2, (R - S1) / 2, (R - S2) / 2, (255, 255, 255))
    elif (R - S1) % 2 != 0 and (R - S2) % 2 == 0:
        im = add_margin(im, (R - S1) / 2 - 0.5, (R - S2) / 2, (R - S1) / 2 + 0.5, (R - S2) / 2, (255, 255, 255))
    elif (R - S1) % 2 == 0 and (R - S2) % 2 != 0:
        im = add_margin(im, (R - S1) / 2, (R - S2) / 2 - 0.5, (R - S1) / 2, (R - S2) / 2 + 0.5, (255, 255, 255))
    else:
        im = add_margin(im, (R - S1) / 2 - 0.5, (R - S2) / 2 - 0.5, (R - S1) / 2 + 0.5, (R - S2) / 2 + 0.5,
                        (255, 255, 255))
    new_fig = tf.Variable(initial_value=tf.constant(0,shape=(1,R,R,S3)),
                          validate_shape=False,shape=(None,R, R, S3),dtype=tf.int32)
    w = rolling_speed  # 1/30 per round
    v = scanning_speed
    f = exposure_rate  # 1/100 rotation rate (smaller means faster exposure time)
    n = 2  # exposure times

    for i in range(R):
        t = i / v
        angle = w * t
        exposure = f
        angle = angle
        new_row = add_exposure(im, angle, exposure, i, n)
        new_fig = new_fig[i,:,:].assign(tf.cast(new_row,dtype=tf.int32))
        # new_fig[i, :, :] = new_row
    # new_fig = Image.fromarray(new_fig)
    return new_fig

