#!/usr/bin/env python

'''
Class Image 2patch, 2image
'''

import os

import imsis as ims

print("Starting...")

fn = r".\images\rice.jpg"
img = ims.Image.load(fn)

overlappx = 0
patches, cols = ims.Image.Tools.image2patches(img, 128, overlappx, verbose=True)

for patch in patches:
    print(patch.shape)

img3 = ims.Image.Tools.patches2image(patches, cols, overlappx, verbose=True)
autoclose=1.2
ims.View.plot(img3, title='Patches to Image', window_title='Patches to Image', autoclose=autoclose)

print("Ready.")
