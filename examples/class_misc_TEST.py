#!/usr/bin/env python

"""
This file is part of IMSIS
Licensed under the MIT license:
http://www.opensource.org/licenses/MIT-license

class misc
"""

import os

import imsis as ims



st = ims.Misc.encodestring("Hello World!")
print(st)
print(ims.Misc.decodestring(st))

print(ims.Misc.uniquetimestamp())

print(ims.Misc.make_filelist(r".\images"))

print(ims.Misc.replacedots("Hello.World! ..."))

itemlist = [[1,2,3],[9,5,7]]
ims.Misc.save_multicolumnlist(r".\output\multicolumn.csv",itemlist,["a","b","c"])
ims.Misc.load_multicolumnlist(r".\output\multicolumn.csv")




