#!/usr/bin/env python

'''
class misc
'''

import os

import imsis as ims



st = ims.Misc.encodestring("Hello World!")
print(st)
print(ims.Misc.decodestring(st))

print(ims.Misc.uniquetimestamp())

print(ims.Misc.make_filelist(r".\images"))

print(ims.Misc.replacedots("Hello.World! ..."))

ims.Misc.list2textfile(r".\output\list.csv",[1,2,3])





