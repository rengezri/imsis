#!/usr/bin/env python

"""
This file is part of IMSIS
Licensed under the MIT license:
http://www.opensource.org/licenses/MIT-license

class analyze test

NOTE: Feature finding and image alignment can be found as seperate examples
"""

import os

import imsis as ims
import cv2 as cv
import numpy as np
import sys

print("Starting...")

# Load and generate test images

fn = r".\images\bberry.jpg"
im_blueberry = ims.Image.load(fn)
fn = r".\images\rice.jpg"
im_rice = ims.Image.load(fn)
fn = r".\images\spa_rice.tif"
im_spa_rice = ims.Image.load(fn)
im_spa_rice = ims.Image.Convert.to8bit(im_spa_rice)
pixelsize = 0  # by default no units
autoclose=1.2


# generated image
w = int(512)
h = int(384)
img5 = np.zeros((h, w, 3), np.uint8)
img5 = cv.circle(img5, (200, 250), radius=25, color=(255, 255, 0), thickness=-1)
img5 = cv.circle(img5, (100, 200), radius=30, color=(128, 128, 0), thickness=-1)
img5 = cv.circle(img5, (220, 50), radius=20, color=(128, 128, 0), thickness=-1)
im_spots = ims.Image.Convert.toGray(img5)

im_blueberry_noise = ims.Image.Process.poisson_noise(im_blueberry)
im_blueberry_shifted = ims.Image.Transform.translate(im_blueberry, 30, 50)
im_rice_gray = ims.Image.Convert.toGray(im_rice)

# DRAWING TEXT AND MEASUREMENTS
img = im_rice_gray

img0 = im_blueberry_noise
img1 = ims.Image.Transform.translate(img0, 15, 23)

img1a = im_blueberry.copy()
img1b = im_blueberry_noise.copy()
img1a = ims.Image.Convert.toRGB(img1a)
img1b = ims.Image.Convert.toRGB(img1b)
img = im_spa_rice


vectors = ims.Dialogs.select_lines(img,"Draw Lines for measurements")


width = 3  # predefined width, which is used to smoothen the signal
for vect in vectors:
    position, length, direction = ims.Analyze.vector_position_length_direction(vect)
    rgb, xpos, ypos = ims.Analyze.find_edge(img, position, width, length, angle=direction, pixelsize=1,
                                        derivative=3,
                                        invert=True, plotresult=True, autoclose=0)


print('Ready.')
