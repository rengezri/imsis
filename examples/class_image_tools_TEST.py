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
img = ims.Image.load(fn)


img2 = ims.Image.Tools.remove_features_at_boundaries(img)
ims.View.plot_list([img,img2])

print("Ready.")
