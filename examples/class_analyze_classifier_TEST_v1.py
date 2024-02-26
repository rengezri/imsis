#!/usr/bin/env python

'''

This file is part of IMSIS
Licensed under the MIT license:
http://www.opensource.org/licenses/MIT-license

class analyze test
'''

import os

import imsis as ims
import csv

print("Starting...")

path_in = r"./images"
path_out = "./output/"

filelist = []
for filename in sorted(os.listdir(path_in)):
    if filename.endswith(('.tiff', '.png', '.tif', '.bmp', '.jpg', '.TIFF', '.PNG', '.TIF', '.BMP', '.JPG')):
        fn = os.path.join(path_in, filename)
        filelist.append(fn)

if not os.path.exists(path_out):
    os.makedirs(path_out)

# Load and generate test images

fn = filelist[2]
im_orig = ims.Image.load(fn)
im_orig_gray = ims.Image.Convert.toGray(im_orig)
im_orig_gray = ims.Image.Convert.to8bit(im_orig_gray)

img = im_orig_gray

user_interaction = True

if user_interaction == True:
    thresh1, min, max, blur = ims.Dialogs.adjust_mask_with_overlay(img, windowtext="Select Mask")
    overlay, minArea, maxArea, dt = ims.Dialogs.adjust_contours_after_masking(img, min, max, blur,
                                                                              windowtext="Adjust Contours")
else:
    min = 0
    max = 128
    blur = 1
    minArea = 200
    maxArea = 1500
    dt = 0.01
    thresh1 = ims.Image.Adjust.thresholdrange(img, min, max)

ims.View.plot(thresh1)

ims.View.plot(ims.Analyze.FeatureClassifier.show_supported_shapes())

detected_objects, img_out = ims.Analyze.FeatureClassifier.classify_features(img, thresh1)
print(detected_objects)
ims.View.plot(img_out)
