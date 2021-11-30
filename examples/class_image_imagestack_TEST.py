#!/usr/bin/env python

'''
Class imagestack test
'''

import imsis as ims

print("Starting...")

fn = r".\images\bberry.jpg"
im_blueberry = ims.Image.load(fn)
im_blueberry_noise = ims.Image.Process.poisson_noise(im_blueberry)

# ANIMATIONS

img0 = im_blueberry_noise
img1 = im_blueberry
s = ims.ImageStack.transition_fadeinout(img0, fadein=False)
ims.ImageStack.play(s)
s = ims.ImageStack.reverse(s)
ims.ImageStack.play(s)
s = ims.ImageStack.transition_dissolve(img0, img1)
ims.ImageStack.play(s)
s = ims.ImageStack.transition_wipe(img0, img1, duration=2 * 15, reverse=False, horizontal=False)
ims.ImageStack.play(s)
s = ims.ImageStack.transition_wipe(img0, img1, duration=2 * 15, reverse=True, horizontal=False)
ims.ImageStack.play(s)
s = ims.ImageStack.scroll(img0, zoomfactor=2, duration=2 * 15, reverse=False, horizontal=True)
ims.ImageStack.play(s)

ims.ImageStack.to_video(s,file_out= r'.\output\video.avi')

print('Ready.')
