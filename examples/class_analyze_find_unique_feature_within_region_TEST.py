"""
This file is part of IMSIS
Licensed under the MIT license:
http://www.opensource.org/licenses/MIT-license

Find feature
"""

import os
import imsis as ims
import numpy as np

print("Starting...")

fn = r".\images\bberry.jpg"
img0 = ims.Image.load(fn)
img0 = ims.Image.Convert.toGray(img0)
img0 = ims.Image.Convert.to16bit(img0)

#dom_col = ims.Analyze.get_dominant_color(img0)
#img0 = ims.Image.Tools.add_border(img0, 50,color=dom_col)  # adding 50 pixels to allow for matching close to borders

# create displaced image
maxdrift = 200
driftx = (np.random.randint(maxdrift) - int(maxdrift / 2))
drifty = (np.random.randint(maxdrift) - int(maxdrift / 2))
driftx, drifty = 35, -23

im3 = ims.Image.Transform.translate(img0, driftx, drifty)

# either select template manually or by code
SHOWDIALOGS = False

find_feature = ims.Analyze.FindFeature()  # create object

if SHOWDIALOGS == True:

    find_feature.create_template_dialog(img0)
    # ims.View.plot(find_feature.template)
else:
    t_center_perc = [0.4, 0.4]
    t_size_perc = [0.25, 0.25]

    t_center_perc = [0.5416666666666666, 0.3828125]
    t_size_perc = [0.2890625, 0.306640625]

    # t_center_perc = [0.5, 0.5]
    # t_size_perc = [0.8, 0.8]

    find_feature.create_template(img0, t_center_perc, t_size_perc)
    # ims.View.plot(find_feature.template)

# define search region
find_feature.set_searchregion_as_template_perc(img0, 2)

# show a plot of the search region and the template
autoclose=1.2
rgb_before = find_feature.plot_searchregion_and_template(img0, verbose=True, autoclose=autoclose)

# im3 = ims.Image.Tools.add_blackmask(im3,[120,50,500,300])
# ims.View.plot(im3)

# run feature finder
find_feature.run(im3, verbose=True)

# plot results
rgb_after = find_feature.plot_matchresult(im3, verbose=False)

autoclose=0
ims.View.plot_list([rgb_before, rgb_after],
                   ["source", "Target=({},{}) Score=({:.2f})".format(find_feature.shift_in_pixels[0], find_feature.shift_in_pixels[1], find_feature.score),
                    "difference"], window_title="Find Feature", autoclose=autoclose)

print("Ready.")
