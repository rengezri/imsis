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

path = os.getcwd()

filename = ims.Dialogs.openfile_dialog(path, 'Open File Dialog', filter="Images (*.png *.jpg *.bmp, *.tif, *.tiff)",
                                       alwaysontop=True)
print(filename)

path = os.getcwd()
filename = ims.Dialogs.savefile_dialog(path, 'Save File Dialog', filter="Images (*.png *.jpg *.bmp, *.tif, *.tiff)",
                                       alwaysontop=True)
print(filename)

items = collections.OrderedDict([('Left', True), ('Right', False), ('Up', True)])  # ordered
print(items)
ret = ims.Dialogs.dialog_checklist(items)
print(ret)

buttons = collections.OrderedDict([('Left', 1), ('Right', 2), ('Up', 3), ('Down', 4), ('Abort', 5)])  # ordered
ret = ims.Dialogs.dialog_multiplebuttons(buttons)
print("button pressed:", ret)

properties = {'Item A': False, 'Item B': True, 'Item C': False}
propertiesafter = ims.Dialogs.dialog_radiobuttonlist(properties)
print(propertiesafter)

properties = {'Item A': False, 'Item B': True, 'Item C': False}
propertiesafter = ims.Dialogs.dialog_comboboxlist(properties)
print(propertiesafter)
ret = ims.Dialogs.dialog_input('fill in a text', "Hello", "Input String")
print(ret, type(ret))
ret = ims.Dialogs.dialog_input('fill in a float', 0, "Input Float")
print(ret, type(ret))


text = "As much text as you like..."
ims.Dialogs.textbox(text, 'Textbox')

text = "As much text as you like..."
ims.Dialogs.textbox_html(text, 'Textbox')

ims.Dialogs.message('This is a message.')
ims.Dialogs.error('An error occured.')

out = ims.Dialogs.dialog_ok_cancel('Would you like to continue?', 'Confirmation Dialog')
print(out)

path = os.getcwd()
folder = ims.Dialogs.openfolder_dialog(path, 'Open Folder Dialog')
print(folder)

properties = {'rows': 3, 'cols': 3, 'enable': True}
propertiesafter = ims.Dialogs.dialog_propertygrid(properties)
print(properties)
print(propertiesafter)

imagelist = ims.Dialogs.dialog_imagelistview("Drag and drop image dialog")
print(imagelist)
# img0 = ims.Image.resize(img, 0.1)


print("Ready.")
