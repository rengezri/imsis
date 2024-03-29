#!/usr/bin/env python

"""
This file is part of IMSIS
Licensed under the MIT license:
http://www.opensource.org/licenses/MIT-license

Class Image test
"""

import os

import imsis as ims
import numpy as np

print("Starting...")

fn = r".\images\bberry.jpg"
im_blueberry = ims.Image.load(fn)
fn = r".\images\rice.jpg"
im_rice = ims.Image.load(fn)
fn = r".\images\spa_rice.tif"
im_spa_rice = ims.Image.load(fn)

im_blueberry_noise = ims.Image.Process.gaussian_noise(im_blueberry)
im_blueberry_shifted = ims.Image.Transform.translate(im_blueberry, 30, 50)
im_rice_gray = ims.Image.Convert.toGray(im_rice)
im_rice_gray_noise = ims.Image.Process.gaussian_noise(im_rice_gray, 0.1)

autoclose = 1.2

info1 = ims.Image.unique_colours(im_blueberry)
info2 = ims.Image.unique_colours(im_rice)
ims.View.plot_list_with_histogram([im_blueberry, im_rice], ['cols {}'.format(info1), 'cols {}'.format(info2)],
                                  autoclose=autoclose)

img2_1 = ims.Image.add(im_blueberry, im_rice, 0.5)
img2_2 = ims.Image.subtract(im_blueberry, im_rice)
ims.View.plot_list([img2_1, img2_2], ['Add', 'Subtract'], window_title='Add / Subtract', autoclose=autoclose)

img2_1 = ims.Image.crop_percentage(im_rice, 0.5)
img2_2 = ims.Image.zoom(im_rice, 1.5, 0.1, 0.1)
img2_3 = ims.Image.resize(ims.Image.Adjust.bin(img2_1, 2), 2)  # first bin than resize to original size
ims.View.plot_list([im_rice, img2_1, img2_2, img2_3], ['Source', 'Crop', 'Zoom', 'Bin'],
                   window_title='Image Crop, Zoom, Bin', autoclose=autoclose)

img2_1 = ims.Image.Transform.flip_vertical(im_rice)
img2_2 = ims.Image.Transform.flip_horizontal(im_rice)
img2_3 = ims.Image.Transform.translate(im_rice, 25, 25)
img2_4 = ims.Image.Transform.rotate(im_rice, 45)
ims.View.plot_list([img2_1, img2_2, img2_3, img2_4], ['Flip vertical', 'Flip horizontal', 'Translate image',
                                                      'Rotate image'], window_title='Image Transformation',
                   autoclose=autoclose)

im2_1 = ims.Image.Process.cannyedge_auto(im_rice)
im2_1 = ims.Image.Binary.morphology_dilate(im2_1, 5)
im_th, im_floodfill, im_floodfill_inv, im_out = ims.Image.Binary.morphology_fillholes(im2_1)
im2_2 = ims.Image.Binary.morphology_erode(im_out, 5)
ims.View.plot_list([im_rice, im2_2], window_title='Image Morphological Filter', autoclose=autoclose)

img2_1 = ims.Image.Binary.morphology_erode(im_rice, 5)
img2_2 = ims.Image.Binary.morphology_dilate(im_rice, 5)
img2_3 = ims.Image.Binary.morphology_open(im_rice, 5)
img2_4 = ims.Image.Binary.morphology_close(im_rice, 5)
ims.View.plot_list([img2_1, img2_2, img2_3, img2_4], ['Erode', 'Dilate', 'Open', 'Close'],
                   window_title='Image Morphological Filter', autoclose=autoclose)

img2_1 = ims.Image.Adjust.thresholdrange(im_rice, 75, 128)
img2_2 = ims.Image.Adjust.threshold(im_rice, 75)
img2_3 = ims.Image.Adjust.invert(im_rice)
ims.View.plot_list([img2_1, img2_2, img2_3], ['Threshold range', 'Threshold binary', 'Invert'],
                   window_title='Image Thresholding, Invert', autoclose=autoclose)

img2_1 = ims.Image.Adjust.histostretch_equalized(im_rice)
img2_2 = ims.Image.Adjust.histostretch_clahe(im_rice)
ims.View.plot_list_with_histogram([im_rice, img2_1, img2_2], ['Source', 'Equalized Histogram', 'Clahe histogram'],
                                  window_title='Image Histogram Optimization', autoclose=autoclose)

img2_1 = im_rice_gray
mx, my, grad, theta = ims.Image.Process.directionalsharpness(img2_1)
ims.View.plot_list([grad, theta], ["Directional sharpness gradient", "Theta"],
                   window_title='Image Directional Sharpness', autoclose=autoclose)



im_4 = ims.Image.Adjust.adjust_auto_whitebalance(im_blueberry)
ims.View.plot_list([im_blueberry, im_4], ['Source', 'Auto whitebalance'], window_title="Auto Whitebalance",
                   autoclose=autoclose)

im_rice_gray_invert = ims.Image.Adjust.invert(im_rice_gray)
img2_1 = ims.Image.Adjust.threshold_otsu(im_rice_gray_invert)
img2_2 = ims.Image.Binary.skeletonize(img2_1)
img2_3 = ims.Image.Binary.zhang_suen_thinning(im_rice_gray)  # very slow
ims.View.plot_list([im_rice, img2_1, img2_2, img2_3], ['Src', 'cannyedge', 'skeletonize', 'zhang_suen_thinning'],
                   window_title='Image Morphological Filter', autoclose=autoclose)

img2_1 = ims.Image.Process.gradient_removal(im_rice_gray, filtersize=513, sigmaX=128)
ims.View.plot_list([im_rice_gray, img2_1], ['Source', 'GradientRemoved'],
                   window_title='Image Edge Enhancement', autoclose=autoclose)

im_rice_small = ims.Image.resize(im_rice, 0.5)
img2_1 = ims.Image.add_overlay(im_rice, im_rice_small)
ims.View.plot_list([im_rice_gray, img2_1], ['Source', 'overlay'],
                   window_title='Overlay', autoclose=autoclose)

print('Ready.')
