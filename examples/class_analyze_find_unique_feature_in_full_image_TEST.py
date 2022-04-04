'''
Template Matching Full Image
'''

import os
import imsis as ims

print("Starting...")

fn = r".\images\bberry.jpg"
im_blueberry = ims.Image.load(fn)

im_blueberry_noise = ims.Image.Process.poisson_noise(im_blueberry)
im_blueberry_shifted = ims.Image.Transform.translate(im_blueberry, 30, 50)

img0 = im_blueberry
img1 = im_blueberry_shifted

img0 = im_blueberry
img1 = im_blueberry_shifted

img3, shiftx, shifty, score = ims.Analyze.ImageAlignment.CenterOfMass(img0, img1)
autoclose=1.2
ims.View.plot_list([img0, img1, img3],
                   ["source", "shifted", "corrected by Center_of_Mass {} {}".format(shiftx, shifty)],
                   window_title='Image Alignment', autoclose=autoclose)

img3, shiftx, shifty, score = ims.Analyze.ImageAlignment.NCC(img0, img1)
ims.View.plot_list([img0, img1, img3], ["source", "shifted", "corrected by NCC {} {}".format(shiftx, shifty)],
                   window_title='Image Alignment', autoclose=autoclose)

img3, shiftx, shifty, score = ims.Analyze.ImageAlignment.GRAD(img0, img1)
ims.View.plot_list([img0, img1, img3], ["source", "shifted", "corrected by GRAD {} {}".format(shiftx, shifty)],
                   window_title='Image Alignment', autoclose=autoclose)

img3, shiftx, shifty, score = ims.Analyze.ImageAlignment.ORB(img0, img1)
ims.View.plot_list([img0, img1, img3], ["source", "shifted", "corrected by ORB {} {}".format(shiftx, shifty)],
                   window_title='Image Alignment', autoclose=autoclose)

img3, shiftx, shifty, score = ims.Analyze.ImageAlignment.NONMAX(img0, img1)
ims.View.plot_list([img0, img1, img3], ["source", "shifted", "corrected by NONMAX {} {}".format(shiftx, shifty)],
                   window_title='Image Alignment', autoclose=autoclose)

print("Ready.")

