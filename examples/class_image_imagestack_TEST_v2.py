#!/usr/bin/env python

"""
This file is part of IMSIS
Licensed under the MIT license:
http://www.opensource.org/licenses/MIT-license

Class imagestack test
"""

import imsis as ims

print("Starting...")

fn = r".\images\bberry.jpg"
im_blueberry = ims.Image.load(fn)
im_blueberry_noise = ims.Image.Process.poisson_noise(im_blueberry)

# ANIMATIONS

img0 = im_blueberry_noise
img1 = im_blueberry

s = ims.ImageStack.create_dummy_imagestack(img0,slices=20,add_noise=0.25,add_translations=0.01,add_rotations=0)
ims.ImageStack.play(s)
t= ims.ImageStack.integrate_images(s)
u= ims.ImageStack.median_images(s)
v=ims.ImageStack.average_images(s)
ims.View.plot_list([t,u,v],["integrate","median","average"])

s = ims.ImageStack.create_dummy_imagestack(img0,slices=20,add_noise=0.25,add_translations=0.05,add_rotations=0)
ims.ImageStack.play(s)
print("calculating shifts and applying shifts found")
t, correctiondatalist = ims.ImageStack.align_images(s,update_reference_image=True,high_precision=False)
ims.ImageStack.play(t)

fn = r".\output\alignmentdata.csv"
ims.Misc.save_multicolumnlist(fn,correctiondatalist,["X","Y","Score"]) #this one is used here, but testing the MISC class
newlist = ims.Misc.load_multicolumnlist(fn)

print("re-applying based on calculated shifts")
u = ims.ImageStack.align_images_reapply(s,correctiondatalist) #re-apply without actual alignment
ims.ImageStack.play(u)

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
s = ims.ImageStack.transition_wipe(img0, img1, duration=2 * 15, reverse=False, horizontal=True)
ims.ImageStack.play(s)
s = ims.ImageStack.transition_wipe(img0, img1, duration=2 * 15, reverse=True, horizontal=True)
ims.ImageStack.play(s)

s = ims.ImageStack.scroll(img0,img1, duration=2 * 15, reverse=False, horizontal=True)
ims.ImageStack.play(s)
s = ims.ImageStack.scroll(img0,img1, duration=2 * 15, reverse=True, horizontal=False)
ims.ImageStack.play(s)
s = ims.ImageStack.scroll(img0,img1, duration=2 * 15, reverse=False, horizontal=True)
ims.ImageStack.play(s)
s = ims.ImageStack.scroll(img0,img1, duration=2 * 15, reverse=True, horizontal=False)
ims.ImageStack.play(s)




ims.ImageStack.to_video(s,file_out= r'.\output\video.avi')

print('Ready.')
