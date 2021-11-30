#!/usr/bin/env python

'''
Class View test
'''


import imsis as ims

print("Starting...")

fn = r".\images\bberry.jpg"
im_blueberry = ims.Image.load(fn)
print(im_blueberry.shape)
print(im_blueberry.dtype)
fn = r".\images\rice.jpg"
im_rice = ims.Image.load(fn)
print(im_rice.shape)
print(im_rice.dtype)
fn = r".\images\spa_rice.tif"
im_spa_rice = ims.Image.load(fn)
print(im_spa_rice.shape)
print(im_spa_rice.dtype)


ims.View.plot(im_blueberry,title="Blueberry",window_title="Single Plot",save_image_filename=r".\output\plot_single_image.png")
ims.View.plot_list([im_blueberry,im_rice,im_spa_rice],titlelist=["Blueberry","Rice","Spaghetti_and_Rice"],window_title="Plot List",save_image_filename=r".\output\Plot_list.png")
ims.View.plot_with_histogram(im_blueberry,title="Blueberry",window_title="Single Plot with Histogram",save_image_filename=r".\output\Plot_with_histogram.png")
ims.View.plot_list_with_histogram([im_blueberry,im_rice],titlelist=["Blueberry","Rice","Spaghetti_and_Rice"],window_title="Plot List with Histogram",save_image_filename=r".\output\Plot_List_with_histogram.png")
ims.View.plot_3dsurface(im_spa_rice,resize=0.15,save_image_filename=r".\output\plot_3dsurface.png")

print('Ready.')
