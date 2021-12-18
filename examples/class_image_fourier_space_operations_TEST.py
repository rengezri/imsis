#!/usr/bin/env python

'''
Class Image test fourier domain
'''

import imsis as ims

print("Starting...")

#fn = r".\images\bberry.jpg"
fn = r".\images\rice.jpg"
#fn = r".\images\spa_rice.tif"

img = ims.Image.load(fn)
img_grey = ims.Image.Convert.to8bit(img)
img_grey = ims.Image.Convert.toGray(img_grey)


img1 = (ims.Image.FilterKernels.ideal_lowpass_kernel(img_grey))
img2 = (ims.Image.FilterKernels.ideal_bandpass_kernel(img_grey))
img3 = (ims.Image.FilterKernels.ideal_bandstop_kernel(img_grey))
img4 = (ims.Image.FilterKernels.gaussian_lowpass_kernel(img_grey))
img5 = (ims.Image.FilterKernels.gaussian_bandpass_kernel(img_grey))
img6 = (ims.Image.FilterKernels.gaussian_bandstop_kernel(img_grey))
img7 = (ims.Image.FilterKernels.butterworth_lowpass_kernel(img_grey))
img8 = (ims.Image.FilterKernels.butterworth_bandpass_kernel(img_grey))
img9 = (ims.Image.FilterKernels.butterworth_bandstop_kernel(img_grey))

ims.View.plot_list([img1, img2, img3, img4, img5, img6, img7, img8, img9],
                   ['idlp', 'idbp', 'idbs', 'gslp', 'gsbp', 'gsbs', 'bwlp', 'bwbp', 'bwbs', ])

filtered, mask, bandcenter, bandwidth, lptype = ims.Dialogs.adjust_FD_bandpass_filter(img)
filtered, mask = ims.Image.Process.FD_bandpass_filter(img, bandcenter, bandwidth,
                                                      bptype=lptype)  # 0=ideal, 1=gaussian, 2=butterworth
ims.View.plot(filtered)

print("Ready.")
