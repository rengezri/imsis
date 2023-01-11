#!/usr/bin/env python

"""
This file is part of IMSIS
Licensed under the MIT license:
http://www.opensource.org/licenses/MIT-license

Class GUI test
"""

import os

import imsis as ims
import collections

fn = r".\images\rice.jpg"
img0 = ims.Image.load(fn)
fn = r".\images\bberry.jpg"
img1 = ims.Image.load(fn)

pixelsize = 1

print("Image adjustment dialogs")

print("compare, use slider to move between images...")
ims.Dialogs.adjust_blending(img0, img1, windowtext="Image Blending")
ims.Dialogs.image_compare(img0, img1)

shapes = ims.Dialogs.select_singlepoint(img0)
shapes = ims.Dialogs.select_points(img0)
shapes = ims.Dialogs.select_lines(img0)
shapes = ims.Dialogs.select_areas(img0)
if (len(shapes) > 0):
    out = ims.Image.crop_rectangle(img0, shapes[0])
    ims.View.plot(out)

out, h, s, v = ims.Dialogs.adjust_HSV(img0)
ims.View.plot(out)
out, h, s, l = ims.Dialogs.adjust_HSL(img0)
ims.View.plot(out)

out, c, b, g = ims.Dialogs.adjust_contrast_brightness_gamma(img0)
ims.View.plot(out)

# the following images need to be in grayscale
img0 = ims.Image.Convert.to8bit(img0)
img0 = ims.Image.Convert.toGray(img0)
img1 = ims.Image.Convert.to8bit(img1)
img1 = ims.Image.Convert.toGray(img1)

out, min, max, blur = ims.Dialogs.adjust_mask(img0)
ims.View.plot(out)
out, min, max, blur = ims.Dialogs.adjust_mask_with_background(img0)
ims.View.plot(out)
out, min, blur = ims.Dialogs.select_edges(img0)
ims.View.plot(out)
out = ims.Dialogs.adjust_openclose(img0)
ims.View.plot(out)

print("Ready.")
