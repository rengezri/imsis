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
import cv2 as cv


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

img3_1 = ims.Image.Colormap.colormap_jet(im_rice)
img3_2 = ims.Image.Colormap.colormap_tab20(im_rice)
img3_3 = ims.Image.Colormap.colormap_hot(im_rice)
img3_4 = ims.Image.Colormap.colormap_tab20b(im_rice)
ims.View.plot_list([img3_1, img3_2, img3_3, img3_4], ['colormap_jet', 'colormap_tab20','colormap_hot', 'colormap_tab20b'], window_title='Image False Colour',
                   autoclose=autoclose)


col = ims.Image.Tools._get_dominant_color(im_rice)
im_colorbar = ims.Image.Tools.draw_color_bar()
img2_2 = ims.Image.Colormap.replace_color_in_colormap(im_colorbar,cv.COLORMAP_JET,[0,0,0])
ims.View.plot_list([im_colorbar, img2_2], ['source','replacecolor'], window_title='Image False Colour',
                   autoclose=autoclose)




print('Ready.')
