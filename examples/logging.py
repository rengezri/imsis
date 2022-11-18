#!/usr/bin/env python

"""
This file is part of IMSIS
Licensed under the MIT license:
http://www.opensource.org/licenses/MIT-license

This module contains methods for logging
"""


from datetime import datetime
import os

class Logging(object):

    @staticmethod
    def log2file(filename, text):
        """Log text to file and console with timestamp.

        :Parameters: path, text
        """

        if (os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename), exist_ok=True)  # mkdir if not empty

        sttime = datetime.now().strftime('%Y%m%d %H:%M:%S,')
        fn = filename
        with open(fn, 'a') as logfile:
            logfile.write(sttime + text + '\n')
        print(sttime+text)



