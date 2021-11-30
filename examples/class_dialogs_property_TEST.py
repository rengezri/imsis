#!/usr/bin/env python

'''
Example dialogs property
'''

import os
import imsis as ims


print("Starting...")

properties = {'Item A': False, 'Item B': True, 'Item C': False}
propertiesafter = ims.Dialogs.dialog_comboboxlist(properties)
print(properties)
print(propertiesafter)

properties = {'Item A': False, 'Item B': True, 'Item C': False}
propertiesafter = ims.Dialogs.dialog_radiobuttonlist(properties)
print(properties)
print(propertiesafter)

properties = {'item_int': 3, 'item_float': 3.5, 'item_bool': True, 'item_string': 'text',
              'item_combo': {'Item A': False, 'Item B': True, 'Item C longer story': False}}

propertiesafter = ims.Dialogs.dialog_propertygrid(properties)
print(properties)
print(propertiesafter)

print("Ready.")
