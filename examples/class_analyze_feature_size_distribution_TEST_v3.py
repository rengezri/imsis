#!/usr/bin/env python

'''
class analyze feature distribution test

'''

import os

import imsis as ims
import csv

print("Starting...")

path_in = r"./images"
path_out = "./output/"

filelist = []
for filename in sorted(os.listdir(path_in)):
    if filename.endswith(('.tiff', '.png', '.tif', '.bmp', '.jpg', '.TIFF', '.PNG', '.TIF', '.BMP', '.JPG')):
        fn = os.path.join(path_in, filename)
        filelist.append(fn)

if not os.path.exists(path_out):
    os.makedirs(path_out)

# Load and generate test images

fn = filelist[1]
im_orig = ims.Image.load(fn)
im_orig_gray = ims.Image.Convert.toGray(im_orig)
im_orig_gray = ims.Image.Convert.to8bit(im_orig_gray)

img = im_orig_gray

user_interaction = True

if user_interaction == True:
    thresh1, min, max, blur = ims.Dialogs.adjust_mask_with_overlay(img, windowtext="Select Mask")
    overlay, minArea, maxArea, dt = ims.Dialogs.adjust_contours_after_masking(img, min, max, blur,
                                                                              windowtext="Adjust Contours")
else:
    min = 96
    max = 255
    blur = 13
    minArea = 200
    maxArea = 1500
    dt = 0.01
    thresh1 = ims.Image.Adjust.thresholdrange(img, min, max)

overlay, labels_original, markers, featurelist = ims.Analyze.FeatureProperties.get_featureproperties(img, thresh1,
                                                                                                     minarea=minArea,
                                                                                                     maxarea=maxArea,
                                                                                                     applydistancemap=True,
                                                                                                     distance_threshold=dt,
                                                                                                     filename=os.path.basename(
                                                                                                         fn),
                                                                                                     colorscheme=ims.Analyze.FeatureProperties.ColorScheme.Original)

overlay, labels_size, markers, featurelist_size = ims.Analyze.FeatureProperties.get_featureproperties(img, thresh1,
                                                                                                      minarea=minArea,
                                                                                                      maxarea=maxArea,
                                                                                                      applydistancemap=True,
                                                                                                      distance_threshold=dt,
                                                                                                      filename=os.path.basename(
                                                                                                          fn),
                                                                                                      colorscheme=ims.Analyze.FeatureProperties.ColorScheme.Size)

overlay, labels, markers, featurelist = ims.Analyze.FeatureProperties.get_featureproperties(img, thresh1,
                                                                                            minarea=minArea,
                                                                                            maxarea=maxArea,
                                                                                            applydistancemap=True,
                                                                                            distance_threshold=dt,
                                                                                            filename=os.path.basename(
                                                                                                fn),
                                                                                            colorscheme=ims.Analyze.FeatureProperties.ColorScheme.Random)

autoclose = 4

im_bbox = ims.Analyze.FeatureProperties.get_image_with_ellipses(labels, featurelist)
unique_timestamp = ims.Misc.uniquetimestamp()

ims.Analyze.FeatureProperties.plot_feature_size_distribution(labels_size, featurelist_size,
                                                             path_out + "FA_{}_size_dist_area".format(
                                                                 unique_timestamp), autoclose=autoclose, num_bins=64,
                                                             sizedistributionxaxis=ims.Analyze.FeatureProperties.SizeDistributionXAxis.Area,plotfigure=False)

ims.Analyze.FeatureProperties.plot_feature_size_distribution(labels_size, featurelist_size,
                                                             path_out + "FA_{}_size_dist_d".format(
                                                                 unique_timestamp), autoclose=autoclose, num_bins=64,
                                                             sizedistributionxaxis=ims.Analyze.FeatureProperties.SizeDistributionXAxis.EquivalentDiameter)

ims.Analyze.FeatureProperties.plot_feature_size_distribution(labels_size, featurelist_size,
                                                             path_out + "FA_{}_size_dist_ar".format(
                                                                 unique_timestamp), autoclose=autoclose, num_bins=64,
                                                             sizedistributionxaxis=ims.Analyze.FeatureProperties.SizeDistributionXAxis.AspectRatio)

ims.Analyze.FeatureProperties.plot_feature_size_ids(labels_size, featurelist_size,
                                                    path_out + "FA_{}_size_ids".format(unique_timestamp),
                                                    autoclose=autoclose)



ims.Analyze.FeatureProperties.save_boundingboxes(labels, featurelist, path_out, max_features_per_page=50)



labels_recolored_original_intensities = labels_original
segmented_image_area = labels_size

original_image = ims.Image.Convert.toRGB(im_orig_gray)

segmented_image_intensity = labels_recolored_original_intensities
labels_intensity_gray = ims.Analyze.FeatureProperties.label_enhance_intensity(segmented_image_intensity)

labels_area_gray = ims.Analyze.FeatureProperties.label_enhance_intensity(segmented_image_area)
labels_area_gray = ims.Analyze.FeatureProperties.label_set_black_background(labels_area_gray)
labels_area_col_jet = ims.Image.Colormap.colormap_jet(labels_area_gray)

labels_intensity_gray = ims.Analyze.FeatureProperties.label_enhance_intensity(segmented_image_intensity)
labels_intensity_gray = ims.Analyze.FeatureProperties.label_set_black_background(labels_intensity_gray)
labels_intensity_col_jet = ims.Image.Colormap.colormap_jet(labels_intensity_gray)

labels_area_overlay = ims.Image.add(original_image, labels_area_col_jet, 0.7)
labels_intensity_overlay = ims.Image.add(original_image, labels_intensity_col_jet, 0.7)
unique_timestamp = ims.Misc.uniquetimestamp()

ims.Image.save(img, path_out + "FA_{}_source.png".format(unique_timestamp))
ims.Image.save(segmented_image_intensity, path_out + "FA_{}_segm_intensity_out.png".format(unique_timestamp))
ims.Image.save(segmented_image_area, path_out + "FA_{}_segm_area_out.png".format(unique_timestamp))
ims.Image.save(labels_area_col_jet, path_out + "FA_{}_segm_area_col_jet.png".format(unique_timestamp))
ims.Image.save(labels_intensity_col_jet, path_out + "FA_{}_segm_intensity_col_jet.png".format(unique_timestamp))

ims.Image.save(labels_area_overlay, path_out + "FA_{}_segm_area_ol.png".format(unique_timestamp))
ims.Image.save(labels_intensity_overlay, path_out + "FA_{}_segm_intensity_ol.png".format(unique_timestamp))

imgstitched = ims.Image.Tools.patches2image(
    [original_image, segmented_image_intensity, labels_area_col_jet, labels_area_overlay],
    cols=2)

ims.Image.save(imgstitched, path_out + "FA_{}_stitched.png".format(unique_timestamp))


ims.Analyze.FeatureProperties.save_featureproperties(path_out + r'FA_{}_feature size distribution.csv'.format(unique_timestamp),
                              featurelist)

featurelist2 = ims.Analyze.FeatureProperties.load_featureproperties(path_out + r'FA_{}_feature size distribution.csv'.format(unique_timestamp))

im_bbox0 = (ims.Analyze.FeatureProperties.get_image_with_boundingboxes(labels, featurelist2))
im_centermarkers = ims.Analyze.FeatureProperties.get_image_with_centermarkers(labels, featurelist2)
ims.View.plot(im_centermarkers, autoclose=autoclose)
ims.View.plot(im_bbox0, autoclose=autoclose)


ims.View.plot_list([img, overlay, labels, im_bbox], ['Source', 'Image Overlay', 'Labels', 'Feature Roundness'],
                   window_title='Feature size distribution', autoclose=autoclose)

ims.View.plot_list([img, labels_intensity_gray, labels_area_col_jet, labels_area_overlay],
                   ['Source', 'Segmented Source Intensity', 'Segmented by Area', 'Segmented by Area Overlay'],
                   window_title='Recoloring', autoclose=autoclose)
