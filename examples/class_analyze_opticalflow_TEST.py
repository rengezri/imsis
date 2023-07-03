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

fn = r".\images\bberry.jpg"
im_blueberry = ims.Image.load(fn)
before = im_blueberry.copy()

imagelist = []
x=0
y=0
for i in range(0,15):
    after = ims.Image.Transform.translate(im_blueberry, x, y)
    hsv, flow, cur_glitch = ims.Analyze.OpticalFlow.dense_optical_flow(before,after)
    imagelist.append(flow)
    x=x+1
    y=y+2

im_blueberry2=after.copy()
for i in range(0,15):
    after = ims.Image.Transform.translate(im_blueberry, x, y)
    hsv, flow, cur_glitch = ims.Analyze.OpticalFlow.dense_optical_flow(before,after)
    imagelist.append(flow)
    x=x-2
    y=y-1

ims.ImageStack.play(imagelist,framerate=5)

