.. imsis documentation master file, created by
   sphinx-quickstart on Wed Jan 23 22:20:13 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.imsis package documentation
===========================================

IMSIS
==================================================

Introduction
------------

IMSIS is an open source image analysis package in python.
The library contains functions to quickly create simple dialog based scripts, fast image processing sequences and perform basic image analysis.
The package relies on powerful libraries such as Numpy, Scipy, OpenCV and QT.

Typical applications would be:

- Dialog based scripts where syntax editing is replaced by runtime dialogs (input dialogs, warnings, property lists, radio button lists, text dialogs etc.)
- Dialog based feature selection (spots, lines, rectangles etc.)
- Fast multi image viewing with or without histograms
- Image batch processing (sharpening, denoising, morphological operations, color operations, image conversion etc.)
- Image analysis (finding unique features, line profiles, counting features, image alignment, image comparisons, image sharpness)
- Image filtering in Fourier space
- Fast image processing for machine learning data

A list of examples of each method can be found in the examples folder.


Requirements
------------

imsis requires the following packages

- numpy 1.13.3
- scipy 1.1.0
- matplotlib 2.0.0
- opencv_python 4.0.0.21
- Pillow 6.2.0
- PyQt5 5.15

Note: versions may change over time.

License
-------

The MIT License (MIT)
Copyright 2021 IMSIS

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


Functions
---------


.. toctree::
   :maxdepth: 2
   :caption: Contents:


imsis Analyze
========================
.. autoclass:: imsis.Analyze
  :members:
  :undoc-members:
  :show-inheritance:

imsis Dialogs
==================
.. autoclass:: imsis.Dialogs(object)
  :members:
  :undoc-members:
  :show-inheritance:

imsis Image
==================
.. autoclass:: imsis.Image
  :members:
  :undoc-members:
  :show-inheritance:

imsis ImageStack
==================
.. autoclass:: imsis.ImageStack
  :members:
  :undoc-members:
  :show-inheritance:

imsis Logging
===================
.. autoclass:: imsis.Logging
  :members:

imsis Misc
===========================
.. autoclass:: imsis.Misc
  :members:
  :undoc-members:
  :show-inheritance:

imsis View
========================
.. autoclass:: imsis.View
  :members:
  :undoc-members:
  :show-inheritance:



Indices and tables
==================

* :ref:`genindex`



