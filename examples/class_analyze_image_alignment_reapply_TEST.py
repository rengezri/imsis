"""
This file is part of IMSIS
Licensed under the MIT license:
http://www.opensource.org/licenses/MIT-license

image alignment
"""

import os
import imsis as ims

path_in0 = r".\images\align/"
alignmentdata = r"E:\ebsdmap\aligned\alignmentdata.csv"
path_out = path_in0 + "/aligned/"

os.makedirs(os.path.dirname(path_out), exist_ok=True)

framelist = ims.ImageStack.load(path_in0)

newlist = ims.Misc.load_multicolumnlist(alignmentdata)
print(newlist)

framelist2 = ims.ImageStack.align_images_reapply(framelist,newlist)
ims.ImageStack.save(framelist2, path_out)

print("Ready.")