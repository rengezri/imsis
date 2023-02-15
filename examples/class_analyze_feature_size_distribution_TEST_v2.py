#!/usr/bin/env python

'''
class analyze feature distribution test

'''

import os

import imsis as ims
import csv

print("Starting...")

path_out = "./output/"

if not os.path.exists(path_out):
    os.makedirs(path_out)

# Load and generate test images

fn = r".\images\rice.jpg"

im_rice = ims.Image.load(fn)
im_rice_gray = ims.Image.Convert.toGray(im_rice)
im_rice_gray = ims.Image.Convert.to8bit(im_rice_gray)
# Feature labeling and counting

img = im_rice_gray

thresh1, min, max, blur = ims.Dialogs.adjust_mask_with_overlay(img, windowtext="Select Mask")
#blur=1
#min=0
#max=163


overlay, minArea,maxArea, dt = ims.Dialogs.adjust_contours_after_masking(img,min,max,blur, windowtext="Adjust Contours")

'''
minArea=200
maxArea=1500
dt=0.2
'''
#ims.View.plot_list([img,overlay,thresh1])


overlay, labels, markers, featurelist = ims.Analyze.FeatureProperties.get_featureproperties(img, thresh1,
                                                                                            minarea=minArea,
                                                                                            maxarea=maxArea,
                                                                                            applydistancemap=True,
                                                                                            distance_threshold=dt,
                                                                                            filename=os.path.basename(fn))
#ims.View.plot_list([overlay,labels])

autoclose = 0
im_bbox = ims.Analyze.FeatureProperties.get_image_with_ellipses(labels, featurelist)

ims.Image.save(markers, path_out + "distancemap.png")
ims.Image.save(thresh1, path_out + "mask.png")
ims.Image.save(labels, path_out + "labels.png")
ims.Image.save(overlay, path_out + "overlay.png")
ims.Image.save(im_bbox, path_out + "bbox.png")

ims.View.plot_list([overlay, labels, im_bbox, markers], ['Image Overlay', 'Labels', 'Boundingboxes', 'Markers'],
                   window_title='Feature size distribution', autoclose=autoclose)

ims.Analyze.FeatureProperties.plot_feature_size_distribution(labels, featurelist, path_out, autoclose=autoclose)

ims.Misc.save_multicolumnlist(path_out + r'feature size distribution.csv', featurelist,
                              ims.Analyze.FeatureProperties.propertynames)

ims.Analyze.FeatureProperties.save_boundingboxes(labels, featurelist, path_out, max_features_per_page=50)

im_bbox0 = (ims.Analyze.FeatureProperties.get_image_with_boundingboxes(labels, featurelist, path_out))
im_centermarkers = ims.Analyze.FeatureProperties.get_image_with_centermarkers(labels, featurelist, path_out)
ims.View.plot(im_centermarkers, autoclose=autoclose)
ims.View.plot(im_bbox0, autoclose=autoclose)
