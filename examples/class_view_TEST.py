#!/usr/bin/env python

"""
This file is part of IMSIS
Licensed under the MIT license:
http://www.opensource.org/licenses/MIT-license

Class View test
"""


import imsis as ims

print("Starting...")

fn = r".\images\bberry.jpg"
im_blueberry = ims.Image.load(fn)
fn = r".\images\rice.jpg"
im_rice = ims.Image.load(fn)
fn = r".\images\spa_rice.tif"
im_spa_rice = ims.Image.load(fn)

autoclose=1.1
ims.View.plot(im_blueberry,title="Blueberry",window_title="Single Plot",save_image_filename=r".\output\plot_single_image.png",autoclose=autoclose)
ims.View.plot_list([im_blueberry,im_rice,im_spa_rice],titlelist=["Blueberry","Rice","Spaghetti_and_Rice"],window_title="Plot List",save_image_filename=r".\output\Plot_list.png",autoclose=autoclose)
ims.View.plot_histogram(im_blueberry,title="Blueberry",window_title="Single Plot with Histogram",save_image_filename=r".\output\Plot_histogram.png",autoclose=autoclose)
ims.View.plot_with_histogram(im_blueberry,title="Blueberry",window_title="Single Plot with Histogram",save_image_filename=r".\output\Plot_with_histogram.png",autoclose=autoclose)
ims.View.plot_list_with_histogram([im_blueberry,im_rice],titlelist=["Blueberry","Rice","Spaghetti_and_Rice"],window_title="Plot List with Histogram",save_image_filename=r".\output\Plot_List_with_histogram.png",autoclose=autoclose)
ims.View.plot_3dsurface(im_spa_rice,resize=0.15,save_image_filename=r".\output\plot_3dsurface.png",autoclose=autoclose)

print('Ready.')
