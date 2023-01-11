#!/usr/bin/env python

"""
This file is part of IMSIS
Licensed under the MIT license:
http://www.opensource.org/licenses/MIT-license

Class Image test
"""

import os

import imsis as ims
import numpy as np

print("Starting...")

fn = r".\images\bberry.jpg"
im_blueberry = ims.Image.load(fn)
fn = r".\images\rice.jpg"
im_rice = ims.Image.load(fn)
fn = r".\images\spa_rice.tif"
im_spa_rice = ims.Image.load(fn)

im_blueberry_noise = ims.Image.Process.gaussian_noise(im_blueberry)
im_blueberry_shifted = ims.Image.Transform.translate(im_blueberry, 30, 50)
im_rice_gray = ims.Image.Convert.toGray(im_rice)
im_rice_gray_noise = ims.Image.Process.gaussian_noise(im_rice_gray, 0.1)




autoclose = 1.2




info1 = ims.Image.unique_colours(im_blueberry)
info2 = ims.Image.unique_colours(im_rice)
ims.View.plot_list_with_histogram([im_blueberry, im_rice], ['cols {}'.format(info1), 'cols {}'.format(info2)],
                                  autoclose=autoclose)

img2_1 = ims.Image.add(im_blueberry, im_rice, 0.5)
img2_2 = ims.Image.subtract(im_blueberry, im_rice)
ims.View.plot_list([img2_1, img2_2], ['Add', 'Subtract'], window_title='Add / Subtract', autoclose=autoclose)

img2_1 = ims.Image.Tools.add_blackmask(im_rice, [50, 50, 250, 250])
ims.View.plot_list([im_rice, img2_1], ['Source', 'With black mask'], window_title='Add Mask', autoclose=autoclose)

img2_1 = ims.Image.Tools.add_blackborder(im_rice, 25)
ims.View.plot_list([im_rice, img2_1], ['Source', 'With black border'], window_title='Add Border', autoclose=autoclose)

img2_1 = ims.Image.crop_percentage(im_rice, 0.5)
img2_2 = ims.Image.zoom(im_rice, 1.5, 0.1, 0.1)
img2_3 = ims.Image.resize(ims.Image.Adjust.bin(img2_1, 2), 2)  # first bin than resize to original size
ims.View.plot_list([im_rice, img2_1, img2_2, img2_3], ['Source', 'Crop', 'Zoom', 'Bin'],
                   window_title='Image Crop, Zoom, Bin', autoclose=autoclose)

img2_1 = ims.Image.Process.poisson_noise(im_rice, 0.25)
img2_2 = ims.Image.Process.gaussian_noise(im_rice, 0.25)
img2_3 = ims.Image.Process.salt_and_pepper_noise(im_rice, 0.25)
ims.View.plot_list([im_rice, img2_1, img2_2, img2_3],
                   ['Source', 'Poisson noise', 'Gaussian noise', 'Salt and pepper noise'],
                   window_title='Image Add Noise', autoclose=autoclose)

img2_1 = ims.Image.Process.gaussian_blur(im_rice, 3)
img2_2 = ims.Image.Process.median(im_rice)
img2_3 = ims.Image.Process.nonlocalmeans(im_rice, h=14, templatewindowsize=9, searchwindowsize=21)
ims.View.plot_list([im_rice, img2_1, img2_2, img2_3], ['Source with noise', 'Gaussianblur', 'Median', 'NonLocalMeans'],
                   window_title='Image Reduce Noise', autoclose=autoclose)

img2_1 = ims.Image.Process.unsharp_mask(im_rice, kernel_size=7, sigma=1.0, amount=1.0, threshold=0)
img2_2 = ims.Image.Process.deconvolution_wiener(im_rice, d=5, noise=11)
ims.View.plot_list([im_rice, img2_1, img2_2], ['Source', 'Unsharpenmask', 'Deconv'], window_title='Image Sharpen',
                   autoclose=autoclose)

img2_1 = ims.Image.Transform.flip_vertical(im_rice)
img2_2 = ims.Image.Transform.flip_horizontal(im_rice)
img2_3 = ims.Image.Transform.translate(im_rice, 25, 25)
img2_4 = ims.Image.Transform.rotate(im_rice, 45)
ims.View.plot_list([img2_1, img2_2, img2_3, img2_4], ['Flip vertical', 'Flip horizontal', 'Translate image',
                                                      'Rotate image'], window_title='Image Transformation',
                   autoclose=autoclose)

img2_1 = ims.Image.Process.cannyedge_auto(im_rice)
img2_2, thetaq = ims.Image.Process.gradient_image_nonmaxsuppressed(img2_1, 5, 40)
img2_3 = ims.Image.Process.pencilsketch((im_rice))
ims.View.plot_list([img2_1, img2_2, img2_3], ['Canny edge auto', 'Gradientnonmaxsupp', 'Pencilsketch'],
                   window_title='Image Edge Enhancement', autoclose=autoclose)

img2_1, angle = ims.Image.Process.gradient_image(im_rice)
ims.View.plot_list([img2_1, angle], ['Gradient_mag', 'Gradient_angle'], window_title='Image Gradient',
                   autoclose=autoclose)



im2_1 = ims.Image.Process.cannyedge_auto(im_rice)
im2_1 = ims.Image.Binary.morphology_dilate(im2_1, 5)
im_th, im_floodfill, im_floodfill_inv, im_out = ims.Image.Binary.morphology_fillholes(im2_1)
im2_2 = ims.Image.Binary.morphology_erode(im_out, 5)
ims.View.plot_list([im_rice, im2_2], window_title='Image Morphological Filter', autoclose=autoclose)

img2_1 = ims.Image.Binary.morphology_erode(im_rice, 5)
img2_2 = ims.Image.Binary.morphology_dilate(im_rice, 5)
img2_3 = ims.Image.Binary.morphology_open(im_rice, 5)
img2_4 = ims.Image.Binary.morphology_close(im_rice, 5)
ims.View.plot_list([img2_1, img2_2, img2_3, img2_4], ['Erode', 'Dilate', 'Open', 'Close'],
                   window_title='Image Morphological Filter', autoclose=autoclose)

img2_1 = ims.Image.Adjust.thresholdrange(im_rice, 75, 128)
img2_2 = ims.Image.Adjust.threshold(im_rice, 75)
img2_3 = ims.Image.Adjust.invert(im_rice)
ims.View.plot_list([img2_1, img2_2, img2_3], ['Threshold range', 'Threshold binary', 'Invert'],
                   window_title='Image Thresholding, Invert', autoclose=autoclose)

img2_1 = ims.Image.Adjust.histostretch_equalized(im_rice)
img2_2 = ims.Image.Adjust.histostretch_clahe(im_rice)
ims.View.plot_list_with_histogram([im_rice, img2_1, img2_2], ['Source', 'Equalized Histogram', 'Clahe histogram'],
                                  window_title='Image Histogram Optimization', autoclose=autoclose)

img3_1 = ims.Image.Process.Falsecolor.falsecolor_jet(im_rice)
img3_2 = ims.Image.Process.Falsecolor.falsecolor_rainbow(im_rice)
ims.View.plot_list([img3_1, img3_2], ['Falsecolor jet', 'Falsecolor rainbow'], window_title='Image False Colour',
                   autoclose=autoclose)

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

img2_1 = im_rice_gray
mx, my, grad, theta = ims.Image.Process.directionalsharpness(img2_1)
ims.View.plot_list([grad, theta], ["Directional sharpness gradient", "Theta"],
                   window_title='Image Directional Sharpness', autoclose=autoclose)

img2_1 = im_blueberry
img2_2 = ims.Image.Transform.rotate(img2_1, 5)
img2_3 = ims.Image.Tools.imageregistration(img2_1, img2_2)
ims.View.plot_list([img2_1, img2_2, img2_3], ['Source', 'Rotated', 'Image Registration of rotated image'],
                   window_title='Image Transformation, Rotation, Registration', autoclose=autoclose)

im_4 = ims.Image.Process.sepia(im_blueberry)
ims.View.plot_list([im_blueberry, im_4], ['Source', 'Sepia'], window_title="Sepia", autoclose=autoclose)

im_4 = ims.Image.Adjust.adjust_auto_whitebalance(im_blueberry)
ims.View.plot_list([im_blueberry, im_4], ['Source', 'Auto whitebalance'], window_title="Auto Whitebalance",
                   autoclose=autoclose)

im_4 = ims.Image.Process.k_means(im_blueberry, k=4)
ims.View.plot_list([im_blueberry, im_4], ['Source', 'K-Means Clustering'], window_title="K-Means Clustering",
                   autoclose=autoclose)

im_4 = ims.Image.Tools.create_hsv_map()
ims.Image.save(im_4 * 255, r'.\output\hsv_map.jpg')
ims.Image.save_withuniquetimestamp(im_4 * 255)

dft_shift, img2_1 = ims.Image.Process.FFT(im_rice)
img2_2 = ims.Image.Process.IFFT(dft_shift)
ims.View.plot_list([im_rice, img2_1, img2_2], ['Source', 'FFT', 'IFFT'], window_title='Image FFT', autoclose=autoclose)


im_rice_gray_invert = ims.Image.Adjust.invert(im_rice_gray)
img2_1 = ims.Image.Adjust.threshold_otsu(im_rice_gray_invert)
img2_2 = ims.Image.Binary.skeletonize(img2_1)
img2_3 = ims.Image.Binary.zhang_suen_thinning(im_rice_gray) #very slow
ims.View.plot_list([im_rice, img2_1,img2_2,img2_3], ['Src', 'cannyedge','skeletonize','zhang_suen_thinning'],
                   window_title='Image Morphological Filter',autoclose=autoclose)

img2_1 = ims.Image.Process.gradient_removal(im_rice_gray,filtersize=513,sigmaX=128)
ims.View.plot_list([im_rice_gray, img2_1], ['Source', 'GradientRemoved'],
                   window_title='Image Edge Enhancement', autoclose=autoclose)




print('Ready.')
