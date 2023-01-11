#!/usr/bin/env python

"""
This file is part of IMSIS
Licensed under the MIT license:
http://www.opensource.org/licenses/MIT-license

Class logging test
"""


import imsis as ims


ims.Logging.log2file("./output/logging.txt","Dummy Text.")

print('Ready.')
