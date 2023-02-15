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

img2_1 = ims.Image.Process.salt_and_pepper_noise(im_rice, 0.35)
img2_2 = ims.Image.Process.replace_black_pixels_by_median(img2_1)
img2_3 = ims.Image.Process.replace_black_pixels_using_nonlocalmeans(img2_1)
ims.View.plot_list([img2_1, img2_2, img2_3],
                   ['Salt and Pepper Noise', 'Remove black pixels by Median', 'Remove black pixels by NLM'],
                   window_title='Image Add Noise', autoclose=autoclose)

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

img2_1 = ims.Image.Process.cannyedge_auto(im_rice)
img2_2, thetaq = ims.Image.Process.gradient_image_nonmaxsuppressed(img2_1, 5, 40)
img2_3 = ims.Image.Process.pencilsketch((im_rice))
ims.View.plot_list([img2_1, img2_2, img2_3], ['Canny edge auto', 'Gradientnonmaxsupp', 'Pencilsketch'],
                   window_title='Image Edge Enhancement', autoclose=autoclose)

img2_1, angle = ims.Image.Process.gradient_image(im_rice)
ims.View.plot_list([img2_1, angle], ['Gradient_mag', 'Gradient_angle'], window_title='Image Gradient',
                   autoclose=autoclose)

img3_1 = ims.Image.Process.Falsecolor.falsecolor_jet(im_rice)
img3_2 = ims.Image.Process.Falsecolor.falsecolor_rainbow(im_rice)
ims.View.plot_list([img3_1, img3_2], ['Falsecolor jet', 'Falsecolor rainbow'], window_title='Image False Colour',
                   autoclose=autoclose)

img2_1 = im_rice_gray
mx, my, grad, theta = ims.Image.Process.directionalsharpness(img2_1)
ims.View.plot_list([grad, theta], ["Directional sharpness gradient", "Theta"],
                   window_title='Image Directional Sharpness', autoclose=autoclose)

im_4 = ims.Image.Process.sepia(im_blueberry)
ims.View.plot_list([im_blueberry, im_4], ['Source', 'Sepia'], window_title="Sepia", autoclose=autoclose)

im_4 = ims.Image.Process.k_means(im_blueberry, k=4)
ims.View.plot_list([im_blueberry, im_4], ['Source', 'K-Means Clustering'], window_title="K-Means Clustering",
                   autoclose=autoclose)

dft_shift, img2_1 = ims.Image.Process.FFT(im_rice)
img2_2 = ims.Image.Process.IFFT(dft_shift)
ims.View.plot_list([im_rice, img2_1, img2_2], ['Source', 'FFT', 'IFFT'], window_title='Image FFT', autoclose=autoclose)

img2_1 = ims.Image.Process.gradient_removal(im_rice_gray, filtersize=513, sigmaX=128)
ims.View.plot_list([im_rice_gray, img2_1], ['Source', 'GradientRemoved'],
                   window_title='Image Edge Enhancement', autoclose=autoclose)

img2_1 = ims.Image.Process.salt_and_pepper_noise(im_rice, 0.05)
col = ims.Analyze.get_dominant_color(im_rice)
img2_1 = ims.Image.Convert.toGray(img2_1)
img2_1 = ims.Image.Binary.morphology_erode(img2_1, 9)
img2_2 = ims.Image.Process.Falsecolor.grayscale_to_color(img2_1, col)
print(ims.Image.info(img2_1))

img2_3 = ims.Image.Process.remove_islands_colour(img2_2, kernel=3)
ims.View.plot_list([img2_1, img2_2, img2_3],
                   ['Salt and Pepper Noise and erode Gray', 'Color', 'Islands Removed'],
                   window_title='Image Add Noise', autoclose=autoclose)

print('Ready.')
