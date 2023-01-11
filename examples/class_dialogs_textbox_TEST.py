#!/usr/bin/env python

"""
This file is part of IMSIS
Licensed under the MIT license:
http://www.opensource.org/licenses/MIT-license

Class GUI test
"""

import os

import imsis as ims
import collections

print("GUI Dialogs")


text = "plain textbox"
ims.Dialogs.textbox(text, 'Textbox')

text = "<b>This is a HTML based textbox</b> <br>"
text = text + "<b>This is normal text</b> <br>"
text = text + r"<b>This is text in bold</b> <br>"
text = text + r"<i>This is text in italic</i> <br>"
text = text + r"<h1 style=""color:blue;"">This is a heading in blue</h1> <br> <p style=""color:red;"">This is a paragraph in red.</p>"

st = r"<figure> <img src=""./images/rice.jpg"" alt=""Rice"" style=""width:50%"">  <figcaption>Fig.1 - Rice</figcaption> </figure>"
text = text +st

ims.Dialogs.textbox_html(text, 'Textbox HTML')


print("Ready.")
