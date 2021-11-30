#!/usr/bin/env python

'''
Class Process test
'''

import os

import imsis as ims
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv

print("Starting...")

fn = r".\images\spa_rice.tif"
img = ims.Image.load(fn)

img = ims.Image.Convert.toGray(img)
print(img.shape, img.dtype)

img = ims.Image.load(fn)
print(img.shape, img.dtype)

img = ims.Image.Convert.to8bit(img)
print(img.shape, img.dtype)

img = ims.Image.Convert.to16bit(img)
print(img.shape, img.dtype)

print("Ready.")

