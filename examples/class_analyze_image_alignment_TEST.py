"""
This file is part of IMSIS
Licensed under the MIT license:
http://www.opensource.org/licenses/MIT-license

image alignment
"""

import os
import imsis as ims

path_in0 = r".\images\align/"
path_out = path_in0 + "/aligned/"
os.makedirs(os.path.dirname(path_out), exist_ok=True)

framelist = ims.ImageStack.load(path_in0)
framelist2, correctiondatalist = ims.ImageStack.align_images(framelist,update_reference_image=True,high_precision=True)

print(correctiondatalist)
ims.ImageStack.save(framelist2, path_out)
ims.Misc.save_multicolumnlist(path_out+"alignmentdata.csv",correctiondatalist,["X","Y","Score"])

newlist = ims.Misc.load_multicolumnlist(path_out+"alignmentdata.csv")
print(newlist)



print("Ready.")