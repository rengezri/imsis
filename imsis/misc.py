#!/usr/bin/env python

"""
This file is part of IMSIS
Licensed under the MIT license:
http://www.opensource.org/licenses/MIT-license

This module contains miscelanous methods
"""

import os
import json
import base64
from datetime import datetime
import re


class Misc(object):
    @staticmethod
    def multipleof2(number):
        """Rounds the given number to the nearest multiple of two.

        :Parameters: float
        :Returns: int
        """
        remainder = number % 2
        if remainder > 1:
            number += (2 - remainder)
        else:
            number -= remainder
        return int(number)

    @staticmethod
    def metadatavaluefromtag(tag, stringobject):
        """Read Metadata value string from image tag

        :Parameters: tag, stringobject
        :Returns: string
        """
        try:
            ms = tag.replace("\r", "").replace("\n", "|")
            regexpst = r".*(" + stringobject + r"=)([A-Za-z0-9-. ]+).*"
            mi = re.match(regexpst, ms)
            txt = str(mi.groups()[1])
        except:
            txt = "-1"
        return txt

    @staticmethod
    def write_json(dct, filename="config.json"):
        """Write JSON dictionary to file

        :Parameters: Filename, dictionary
        """
        with open(filename, 'w') as fp:
            json.dump(dct, fp, sort_keys=True, indent=4)

    @staticmethod
    def read_json(filename="config.json"):
        """Read JSON dictionary from file

        :Parameters: Filename, dictionary
        :Returns: Dictionary

        """
        with open(filename, 'r') as fp:
            dct = json.load(fp)
        return dct

    @staticmethod
    def replacedots(st):
        """Replace all dots in a string

        :Parameters: string
        :Returns: string
        """
        for char in st:
            if char in ".":
                st_new = st.replace(char, '_')
        return st_new

    @staticmethod
    def make_filelist(path_in0):
        """Make filelist of all images in path

        :Parameters:  path_in
        :Returns: file list
        """

        fn_list = []
        for file in sorted(os.listdir(path_in0)):
            if file.endswith(('.tiff', '.png', '.tif', '.bmp', '.jpg', '.TIFF', '.PNG', '.TIF', '.BMP', '.JPG')):
                fn_list.append(os.path.join(path_in0, file))
        return fn_list

    @staticmethod
    def uniquetimestamp():
        """Create a unique timestamp

        :Returns: string
        """
        sttime = datetime.now().strftime('%Y%m%d%H%M%S')
        return sttime

    @staticmethod
    def encodestring(input_string):
        """encode a string
        This routine converts a string to a sequence of bytes
        Text will become obfuscated not encrypted.

        :Returns: string
        """

        encoded = base64.b64encode(bytes(input_string, "utf-8"))
        return encoded

    @staticmethod
    def decodestring(input_string):
        """decode a string
        This routine converts a sequence of bytes to a string
        Text will become obfuscated not encrypted.

        :Returns: string
        """
        pw = base64.b64decode(input_string).decode("utf-8", "ignore")
        return pw

    @staticmethod
    def multicolumnlist2textfile(filename, itemlist):
        """Save string to textfile
        DEPRECIATED

        :Parameters: path, itemlist
        """

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "w") as outfile:
            outfile.write("\n".join(str(item).lstrip('[').rstrip(']') for item in itemlist))

    @staticmethod
    def save_multicolumnlist(filename, itemlist, headerlist):
        """Save list with array of string to each line
        add header with properties if enabled

        :Parameters: path, itemlist, header
        """

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "w") as outfile:
            if headerlist:
                outfile.write(','.join(str(item) for item in headerlist) + "\n")
            outfile.write("\n".join(str(item).lstrip('[').rstrip(']') for item in itemlist))

    @staticmethod
    def load_multicolumnlist(filename):
        """load list with array of string to each line
        header will be ignored (mainly for reading in notepad/excel etc.)

        :Parameters: filename
        :Returns: list
        """

        if os.path.exists(filename):

            data = []
            with open(filename) as f:
                first_row = True
                for line in f:
                    if first_row:
                        first_row = False
                        continue
                    columns = line.strip().split(",")
                    converted_values = []
                    for val in columns:
                        converted_val = None
                        if val.lower() == "true":
                            converted_val = True
                        elif val.lower() == "false":
                            converted_val = False
                        else:
                            try:
                                converted_val = int(val)
                            except ValueError:
                                try:
                                    converted_val = float(val)
                                except ValueError:
                                    converted_val = val
                        converted_values.append(converted_val)
                    data.append(converted_values)
        else:
            print("Warning: multicolumn list file does not exist.")
            data = None
        return data

    @staticmethod
    def obj2dict(obj, verbose=False):
        """Convert Object to Dictionary

        can be used for json serialization of objects in a class

        :Parameters: Object
        :Returns: Dictionary
        """
        pr = {}
        for name in dir(obj):
            value = getattr(obj, name)
            if not name.startswith('__') and not inspect.ismethod(value):
                if (type(value) == dict):
                    pr[name] = "dict({0})".format(value)
                    if (verbose == True):
                        print(str(value))
                else:
                    pr[name] = value
        return pr

    @staticmethod
    def dict2obj(src_dct, tar_obj):
        """Convert Dictionary to predefined Object

        can be used for json serialization of objects in a class

        :Parameters: Dictionary
        :Returns: Target Object
        """
        for k, v in src_dct.items():
            setattr(tar_obj, k, v)
            if (type(v) == str):
                if (v.startswith("dict")):
                    v = v.split("(", 1)[1]  # remove dict(
                    v = v[:-1]  # remove last )
                    v2 = eval(v)
                    setattr(tar_obj, k, v2)
                else:
                    setattr(tar_obj, k, v)
        return tar_obj
