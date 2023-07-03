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

# Load and generate test images

fn = r".\images\bberry.jpg"
im_blueberry = ims.Image.load(fn)
fn = r".\images\rice.jpg"
im_rice = ims.Image.load(fn)
fn = r".\images\spa_rice.tif"
im_spa_rice = ims.Image.load(fn)
im_spa_rice = ims.Image.Convert.to8bit(im_spa_rice)
pixelsize = 0  # by default no units
autoclose=1.2


# generated image
w = int(512)
h = int(384)
img5 = np.zeros((h, w, 3), np.uint8)
img5 = cv.circle(img5, (200, 250), radius=25, color=(255, 255, 0), thickness=-1)
img5 = cv.circle(img5, (100, 200), radius=30, color=(128, 128, 0), thickness=-1)
img5 = cv.circle(img5, (220, 50), radius=20, color=(128, 128, 0), thickness=-1)
im_spots = ims.Image.Convert.toGray(img5)

im_blueberry_noise = ims.Image.Process.poisson_noise(im_blueberry)
im_blueberry_shifted = ims.Image.Transform.translate(im_blueberry, 30, 50)
im_rice_gray = ims.Image.Convert.toGray(im_rice)

# DRAWING TEXT AND MEASUREMENTS
img = im_rice_gray

img0 = im_blueberry_noise
img1 = ims.Image.Transform.translate(img0, 15, 23)

img1a = im_blueberry.copy()
img1b = im_blueberry_noise.copy()
img1a = ims.Image.Convert.toRGB(img1a)
img1b = ims.Image.Convert.toRGB(img1b)

img4 = ims.Image.Process.cannyedge_auto(im_rice, 0.33)
img4b, lines = ims.Analyze.hough_lines(img4, threshold=2, minlinelength=50, maxlinegap=5)
ims.View.plot_list([im_rice, img4, img4b], ['Source', "CannyEdge", 'Hough Lines'], window_title="Hough Lines",autoclose=autoclose)

image_identical = ims.Analyze.compare_image_identical(img1a, img1b)
image_mse = ims.Analyze.compare_image_mse(img1a, img1b)
print("Image A=B ? :", image_identical)
print("Image A vs B MSE:", image_mse)
image_identical = ims.Analyze.compare_image_identical(img1a, img1a)
image_mse = ims.Analyze.compare_image_mse(img1a, img1a)
print("Image A=C ? :", image_identical)
print("Image A vs C MSE:", image_mse)
ims.View.plot_list([img1a, img1b, img1a], ['Image A', 'Image B (noise added)', 'Image C'],
                   window_title="Image Comparison",autoclose=autoclose)

image_empty = ims.Analyze.is_image_empty(img1a)
print("Is image empty? ", image_empty)

# LINE MEASUREMENTS
vectors = [[(38, 14), (38, 80)]]
width = 40  # predefined width, which is used to smoothen the signal
print(vectors[0])
position, length, direction = ims.Analyze.vector_position_length_direction(vectors[0])

img = im_spa_rice
x, y = ims.Analyze.get_lineprofile(img, position, length, width, direction, pixelsize=1,autoclose=autoclose)
rgb, xpos, ypos = ims.Analyze.find_edge(img, position, width, length, angle=direction, pixelsize=1,
                                        derivative=1,
                                        invert=True, plotresult=True, autoclose=autoclose)

vectors = [[(200, 205), (200, 293)], [(182, 51), (256, 52)]]
position, length, direction = ims.Analyze.vector_position_length_direction(vectors[0])

img = im_spots

img4, widthout = ims.Analyze.measure_linewidth(img, position, width, length, direction, pixelsize=pixelsize,
                                               derivative=1,
                                               linethickness=1, invert=True,
                                               plotresult=True, plotboundingbox=True, autoclose=autoclose)

img6, results = ims.Analyze.measure_lines(img, vectors, linewidth=50, pixelsize=pixelsize, derivative=1, invert=True,
                                          verbose=False, autoclose=autoclose)

img = im_rice_gray
# DRAWING TEXT AND MEASUREMENTS
# ims.Dialogs.select_lines(img)
pixelsize2 = (55. / 1000) / img.shape[1]
vectors = [[(227, 131), (249, 82)], [(211, 200), (270, 190)]]

img2_1, results = ims.Analyze.add_line_measurements(img, vectors, pixelsize=pixelsize2, fontsize=30, verbose=False)
img2_1 = ims.Analyze.add_text(img2_1, int(img.shape[1] * 0.5), int(img.shape[0] * 0.05), 'Rice grains', 20,
                              aligntocenter=True, outline=True)
img2_1 = ims.Analyze.add_scalebar(img2_1, pixelsize2)

ims.View.plot(img2_1, '',
              window_title="Draw measurements on image",autoclose=autoclose)

# FINDING SPOTS

img5 = im_spots

img3, dx, dy = ims.Analyze.find_brightest_spot(img5, pixelsize=1)
ims.View.plot_list([img5, img3], ["Source", "Brightest spot pos=({},{})".format(dx, dy)], window_title="Find Brightest Spot",autoclose=autoclose)

img3, dx, dy = ims.Analyze.find_contour_center(img5)
ims.View.plot_list([img5, img3], ["Source", "Contour center pos=({},{})".format(dx, dy)], window_title="Find Contour Center",autoclose=autoclose)

img3, dx, dy = ims.Analyze.find_image_center_of_mass(img5)
ims.View.plot_list([img5, img3], ["Source", "Center of mass pos=({},{})".format(dx, dy)],
                   window_title="Find Image Center of Mass",autoclose=autoclose)

img2 = ims.Image.Adjust.thresholdrange(img, 3, 64)
img2, mlist = ims.Analyze.measure_spheres(img, img2, pixelsize, areamin=50, areamax=100000)

print("Combinations of classes")

img0 = im_blueberry_noise

img0 = img
shiftx = 5
shifty = 5
img1 = ims.Image.Transform.translate(img0, shiftx, shifty)
img1 = ims.Image.Process.poisson_noise(img1)
ims.View.plot_list([img0, img1], ["Source", "Destination pos=({},{})".format(shiftx, shifty)],autoclose=autoclose)

# SHARPNESS MEASUREMENTS

sharpness = ims.Analyze.SharpnessDetection()

print("LAPV: ", sharpness.varianceOfLaplacian(img1))
print("TENG: ", sharpness.tenengrad(img1))
print("TENV: ", sharpness.tenengradvariance(img1))
print("GLVN: ", sharpness.normalizedGraylevelVariance(img1))
print("GLVA: ", sharpness.graylevelVariance(img1))
print("LAPM: ", sharpness.modifiedLaplacian(img1))
print("LAPD: ", sharpness.diagonalLaplacian(img1))
print("CURV: ", sharpness.curvature(img1))
print("BREN: ", sharpness.brenner(img1))
print("SMD: ", sharpness.sumModulusDifference(img1))
print("SMD2: ", sharpness.sumModulusDifference2(img1))
print("EGR: ", sharpness.energygradient(img1))
print("VCR: ", sharpness.vollathsautocorrelation(img1))
print("EHS: ", sharpness.entropy(img1))

print("PSNR: ", ims.Analyze.PSNR(img1, img0))

ims.Analyze.powerspectrum(im_blueberry_noise, autoclose=autoclose)

print('Ready.')
