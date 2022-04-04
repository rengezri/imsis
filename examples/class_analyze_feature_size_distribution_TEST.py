#!/usr/bin/env python

'''
class analyze feature distribution test

'''

import os

import imsis as ims

print("Starting...")

# Load and generate test images
fn = r".\images\rice.jpg"
im_rice = ims.Image.load(fn)
im_rice_gray = ims.Image.Convert.toGray(im_rice)
im_rice_gray = ims.Image.Convert.to8bit(im_rice_gray)


# Feature labeling and counting

img = im_rice_gray

# thresh1, min, max, blur = ims.Dialogs.adjust_mask(img)
thresh1 = ims.Image.Process.gaussian_blur(img, 3)
thresh1 = ims.Image.Adjust.thresholdrange(thresh1, 86, 147)
overlay, out, cntsval,sizedistout = ims.Analyze.feature_size_distribution(img, thresh1)
# ims.View.plot(overlay, '')
# ims.View.plot(out, '')

autoclose=2
ims.View.plot_list([overlay, out], ['overlay', 'out'], window_title='Feature size distribution', autoclose=autoclose)
print(sizedistout)
ims.Misc.multicolumnlist2textfile(r'.\output\feature size distribution.csv', sizedistout)

print("Ready.")
