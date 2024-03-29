# IMSIS

## Introduction

IMSIS is an open source image analysis package in Python. The library contains functions to quickly create simple dialog-based scripts, fast image processing sequences, and perform basic image analysis. The package relies on powerful libraries such as Numpy, Scipy, OpenCV, and PySide6.

Typical applications would be:
- Dialog-based scripts where syntax editing is replaced by runtime dialogs (input dialogs, warnings, property lists, radio button lists, text dialogs, etc.)
- Dialog-based feature selection (spots, lines, rectangles, etc.)
- Fast multi-image viewing with or without histograms
- Image batch processing (sharpening, denoising, morphological operations, color operations, image conversion, etc.)
- Image analysis (finding unique features, line profiles, counting features, image alignment, image comparisons, image sharpness)
- Image filtering in Fourier space
- Fast image processing for machine learning data

## Requirements

As in requirements file. Python > 3.5

## Requirements Documentation

IMSIS Documentation building requires the following additional packages:
- sphinx 5.3.0 (Python 3.10)
- sphinx_rtd_theme 1.1.1 (Python 3.10)
- rinohtype 0.5.4 (Python 3.10)

Documentation can be automatically generated with `python build_docs.py`.

## Installation

To install IMSIS, run the following commands:

```bash
python setup.py sdist bdist_wheel
pip install dist\imsis-1.0-py3-none-any.whl
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.


Example
-------

A simple example of loading and displaying an image

```
import imsis as ims

fn = r".\images\bberry.jpg"
im_blueberry = ims.Image.load(fn)

ims.View.plot(im_blueberry,title="Blueberry",window_title="Plot")
```

A list of examples of every method implemented can be found in the examples folder.

Some more can be found below:


Animated transitions
--------------------
<img src="./figures/animated_transitions.jpg" width="300">

Image blending
--------------
<img src="./figures/blending.jpg" width="300">

Image denoising
---------------
<img src="./figures/denoise.jpg" width="300">

Interactive user dialogs
------------------------
<img src="./figures/dialogs.jpg" width="300">

Measurements on images
----------------------
<img src="./figures/measurements_on_image.jpg" width="300">

Feature descriptor Matching
---------------------------
<img src="./figures/feature_descriptor_matching.jpg" width="300">

Find Brightest Spot
-------------------
<img src="./figures/find_brightest_spot.jpg" width="300">

Find Edges
----------
<img src="./figures/find_edges.jpg" width="300">

Find Feature
------------
<img src="./figures/find_feature.jpg" width="300">

Frequency domain image filtering
--------------------------------
<img src="./figures/frequency_domain_filtering.jpg" width="300">

Histogram operations
--------------------
<img src="./figures/histogram_operations.jpg" width="300">

HSV color channel editing
-------------------------
<img src="./figures/hsv_Channels.jpg" width="300">

K-means clustering
------------------
<img src="./figures/k_meansclustering.jpg" width="300">

Image masking
-------------
<img src="./figures/masking.jpg" width="300">
