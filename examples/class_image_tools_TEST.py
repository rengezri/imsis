#!/usr/bin/env python

"""
This file is part of IMSIS
Licensed under the MIT license:
http://www.opensource.org/licenses/MIT-license

Class Image 2patch, 2image
"""

import os

import imsis as ims
import cv2
import numpy as np

print("Starting...")

fn = r".\images\bberry.jpg"
im_blueberry = ims.Image.load(fn)
fn = r".\images\rice.jpg"
im_rice = ims.Image.load(fn)
fn = r".\images\spa_rice.tif"
im_spa_rice = ims.Image.load(fn)

autoclose=5
img = im_blueberry

images = ims.ImageStack.create_dummy_imagestack(im_blueberry, slices=20, add_noise=0.25, add_translations=0.01, add_rotations=0)
out = ims.Image.Tools.create_focus_stack_simple(images)
ims.View.plot_list([images[0],out],['source','focusstack'], autoclose=autoclose)

img2 = ims.Image.Tools.remove_features_at_boundaries(im_blueberry)
ims.View.plot_list([img,img2])

img2_1 = ims.Image.Tools.add_blackmask(im_rice, [50, 50, 250, 250])
ims.View.plot_list([im_rice, img2_1], ['Source', 'With black mask'], window_title='Add Mask', autoclose=autoclose)

img2_1 = ims.Image.Tools.add_blackborder(im_rice, 25)
ims.View.plot_list([im_rice, img2_1], ['Source', 'With black border'], window_title='Add Border', autoclose=autoclose)


img3_1 = ims.Image.Tools.create_checkerboard()
img3_2 = ims.Image.Tools.image_with_2_closeups(im_rice)
img3_3 = ims.Image.Tools.squared(im_rice)
ims.View.plot_list([img3_1, img3_2, img3_3], ['Checkerboard', '2close-ups', 'Squared'], window_title='Image Misc',
                   autoclose=autoclose)

# fisheye
K = np.array(
    [[781.3524863867165, 0.0, 794.7118000552183], [0.0, 779.5071163774452, 561.3314451453386], [0.0, 0.0, 1.0]])
fx = 1200
fy = 1200

K = np.array(
    [[fx, 0.0, img3_1.shape[0] * 0.5], [0.0, fy, img3_1.shape[1] * 0.5], [0.0, 0.0, 1.0]])
D = np.array([[0.0], [0.0], [0.0], [1.0]])
img1_1 = ims.Image.Tools.fisheye_correction(img3_1, K, D, DIM=(img3_1.shape[0], img3_1.shape[1]))
ims.View.plot_list([img3_1, img1_1], ["Source", "Fisheye"], window_title='Image Fisheye', autoclose=autoclose)

img2_1 = im_blueberry
img2_2 = ims.Image.Transform.rotate(img2_1, 5)
img2_3 = ims.Image.Tools.imageregistration(img2_1, img2_2)
ims.View.plot_list([img2_1, img2_2, img2_3], ['Source', 'Rotated', 'Image Registration of rotated image'],
                   window_title='Image Transformation, Rotation, Registration', autoclose=autoclose)


im_4 = ims.Image.Tools.create_hsv_map()
ims.Image.save(im_4 * 255, r'.\output\hsv_map.jpg')
ims.Image.save_withuniquetimestamp(im_4 * 255)



print("Ready.")
