#!/usr/bin/env python

"""
This file is part of IMSIS
Licensed under the MIT license:
http://www.opensource.org/licenses/MIT-license


Example dialogs property
"""

import os
import imsis as ims
import collections

print("Starting...")
properties = {'Item A': False, 'Item B': True, 'Item C': False}
propertiesafter = ims.Dialogs.dialog_comboboxlist(properties)
print(properties)
print(propertiesafter)
print('out: ', ims.Dialogs.dialog_dictionary_activekey(propertiesafter))

properties = {'Item A': False, 'Item B': True, 'Item C': False}
propertiesafter = ims.Dialogs.dialog_radiobuttonlist(properties)
print(properties)
print(propertiesafter)
print('out: ', ims.Dialogs.dialog_dictionary_activekey(propertiesafter))

properties = {'item_int': 3, 'item_float': 3.5, 'item_bool': True, 'item_string': 'text',
              'item_combo': {'Item A': False, 'Item B': False, 'Item C longer story': True},
              'item_string_long': 'more text', 'item_combo2': {'Item A2': True, 'Item B2': False}, 'item_bool2': True}

propertiesafter = ims.Dialogs.dialog_propertygrid(properties)
print(properties)
print(propertiesafter)
print(ims.Dialogs.dialog_dictionary_activekey(properties['item_combo']))
print(ims.Dialogs.dialog_dictionary_activekey(propertiesafter['item_combo']))

print("Ready.")
