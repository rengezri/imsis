#!/usr/bin/env python

'''
Class logging test
'''


import imsis as ims


ims.Logging.log2file("./output/logging.txt","Dummy Text.")

print('Ready.')
