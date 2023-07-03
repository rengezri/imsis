#!/usr/bin/env python

"""
This file is part of IMSIS
Licensed under the MIT license:
http://www.opensource.org/licenses/MIT-license

This module contains methods to display images
"""

import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import pyplot as plt
import imsis as ims
import os


class View(object):

    @staticmethod
    def plot(img, title='', window_title='Plot', save_image_filename="",autoclose=0):
        """Plot a single image with title

        :Parameters: image, title, window_title, save_image_filename
        """
        # img = cv.pyrDown(img)

        try:
            dummy = (img.shape)
        except:
            print('Error, image shape mismatch found in plot.')

        plt.figure(figsize=(8, 8))
        try:
            plt.gcf().canvas.set_window_title(window_title)
        except:
            plt.gcf().canvas.setWindowTitle(window_title)

        plt.title(title)

        img = ims.Image.Convert.BGRtoRGB(img)

        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        if save_image_filename:
            os.makedirs(os.path.dirname(save_image_filename), exist_ok=True)
            plt.savefig(save_image_filename)  # save the figure to file

        if autoclose == -1:
            plt.close("all")
        else:
            if autoclose > 0:
                try:
                    plt.show(block=False)
                    plt.pause(autoclose)  # 3 seconds, I use 1 usually
                except:
                    print("Interrupted while waiting.")
                plt.close("all")
            else:
                plt.show()
            plt.close()

    @staticmethod
    def plot_list(imglist, titlelist=[''], window_title='Plot', save_image_filename="",autoclose=0):
        """Plot list of images and list of titles in single overview

        :Parameters: imagelist, titlelist, window_title, save_image_filename
        """

        # replace titlelist by emptylist if list is shorter than imagelist
        if len(titlelist) < len(imglist):
            titlelist = []
            for i in range(len(imglist)):
                titlelist.append("")

        ln = len(imglist)
        rows = int(np.math.sqrt(ln))
        cols = int(ln / rows + 0.5)

        plt.figure(figsize=(cols * 4, rows * 4))

        try:
            plt.gcf().canvas.set_window_title(window_title)
        except:
            plt.gcf().canvas.setWindowTitle(window_title)

        # print(rows,cols)
        i = 0
        j = 0
        m = 0

        for img in imglist:
            try:
                sh = img.shape
            except:
                print('Error, image shape mismatch in plot list.')

            img0 = ims.Image.Convert.BGRtoRGB(img)
            title = titlelist[m]
            n = int("{0}{1}{2}".format(rows, cols, m + 1))
            # print(n)
            # plt.subplot(n), plt.imshow(img0, cmap='gray')
            plt.subplot(n)
            if len(img0.shape) == 2:
                plt.imshow(img0, cmap='gray')
            else:
                plt.imshow(img0)
            plt.title(title)
            i = i + 1
            if (i > rows):
                i = 0
                j = j + 1
            m = m + 1
        if save_image_filename:
            os.makedirs(os.path.dirname(save_image_filename), exist_ok=True)
            plt.savefig(save_image_filename)  # save the figure to file
        if autoclose == -1:
            plt.close("all")
        else:
            if autoclose > 0:
                try:
                    plt.show(block=False)
                    plt.pause(autoclose)  # 3 seconds, I use 1 usually
                except:
                    print("Interrupted while waiting.")
                plt.close("all")
            else:
                plt.show()
            plt.close()

    # cdf cumulative distribution function
    # plot image and histogram
    @staticmethod
    def plot_with_histogram(img, title='', window_title='Plot', save_image_filename="",autoclose=0):
        """Plot a single image with title and histogram

        :Parameters: image, title, window_title, save_image_filename
        """
        # w, h = img.shape[::-1]

        # img = Image.Convert.BGRtoRGB(img)

        w = img.shape[1]
        h = img.shape[0]
        img = ims.Image.Convert.BGRtoRGB(img)

        if (img.dtype == np.uint8):
            rng = 256
        else:
            rng = 65535
        # bitdepth = img.dtype
        hist, bins = np.histogram(img.flatten(), 256, [0, rng])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()  # this line not necessary.
        plt.figure(figsize=(8, 8))

        try:
            plt.gcf().canvas.set_window_title(window_title)
        except:
            plt.gcf().canvas.setWindowTitle(window_title)

        gridspec.GridSpec(3, 1)
        plt.subplot2grid((3, 1), (0, 0), colspan=1, rowspan=2)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.title(title)
        plt.subplot2grid((3, 1), (2, 0), colspan=1, rowspan=1)
        plt.plot(cdf_normalized, color='b')
        # plt.hist(img.flatten(), bitdepth, [0, bitdepth], color='0.30')
        plt.hist(img.flatten(), 256, [0, rng], color='0.30')
        plt.xlim([0, rng])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.tight_layout()
        if save_image_filename:
            os.makedirs(os.path.dirname(save_image_filename), exist_ok=True)
            plt.savefig(save_image_filename)  # save the figure to file
        if autoclose == -1:
            plt.close("all")
        else:
            if autoclose > 0:
                try:
                    plt.show(block=False)
                    plt.pause(autoclose)  # 3 seconds, I use 1 usually
                except:
                    print("Interrupted while waiting.")
                plt.close("all")
            else:
                plt.show()
            plt.close()

    # cdf cumulative distribution function
    # plot image and histogram
    @staticmethod
    def plot_histogram(img, title='', window_title='Plot', save_image_filename="",autoclose=0):
        """Plot a histogram (no image)

        :Parameters: image, title, window_title, save_image_filename
        """
        # w, h = img.shape[::-1]

        # img = Image.Convert.BGRtoRGB(img)

        w = img.shape[1]
        h = img.shape[0]
        img = ims.Image.Convert.BGRtoRGB(img)

        if (img.dtype == np.uint8):
            rng = 256
        else:
            rng = 65535
        # bitdepth = img.dtype
        hist, bins = np.histogram(img.flatten(), 256, [0, rng])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()  # this line not necessary.
        plt.figure(figsize=(8, 8))

        try:
            plt.gcf().canvas.set_window_title(window_title)
        except:
            plt.gcf().canvas.setWindowTitle(window_title)

        gridspec.GridSpec(1, 1)
        plt.title(title)
        plt.plot(cdf_normalized, color='b')
        # plt.hist(img.flatten(), bitdepth, [0, bitdepth], color='0.30')
        plt.hist(img.flatten(), 256, [0, rng], color='0.30')
        plt.xlim([0, rng])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.tight_layout()
        if save_image_filename:
            os.makedirs(os.path.dirname(save_image_filename), exist_ok=True)
            plt.savefig(save_image_filename)  # save the figure to file
        if autoclose == -1:
            plt.close("all")
        else:
            if autoclose > 0:
                try:
                    plt.show(block=False)
                    plt.pause(autoclose)  # 3 seconds, I use 1 usually
                except:
                    print("Interrupted while waiting.")
                plt.close("all")
            else:
                plt.show()
            plt.close()



    @staticmethod
    def plot_list_with_histogram(imglist, titlelist=[''], window_title='Plot', save_image_filename="",autoclose=0):
        """Plot list of images and list of titles in single overview with histogram

        :Parameters: imagelist, titlelist, window_title, save_image_filename
        """

        # replace titlelist by emptylist if list is shorter than imagelist
        if len(titlelist) < len(imglist):
            titlelist = []
            for i in range(len(imglist)):
                titlelist.append("")

        ln = len(imglist)
        rows = int(np.math.sqrt(ln))
        cols = int(ln / rows + 0.5)

        plt.figure(figsize=((cols + 1) * 2, rows * 4))

        # plt.figure(figsize=((cols+1)*2,rows*4))
        try:
            plt.gcf().canvas.set_window_title(window_title)
        except:
            plt.gcf().canvas.setWindowTitle(window_title)

        gridspec.GridSpec(2, cols)
        # print('gridspec figsize',2,cols,cols*8,8)

        # print(rows,cols)
        i = 0
        j = 0
        m = 0
        for img in imglist:
            img0 = ims.Image.Convert.BGRtoRGB(img)
            if (img0.dtype == np.uint8):
                rng = 256
            else:
                rng = 65535

            hist, bins = np.histogram(img.flatten(), 256, [0, rng])
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max()  # this line not necessary.

            title = titlelist[m]
            n = int("{0}{1}{2}".format(rows, cols, m + 1))
            # print(n)
            # plt.subplot(n), plt.imshow(img0, cmap='gray')

            # print(cols,m)
            plt.subplot2grid((2, cols), (0, m), colspan=1, rowspan=1)
            plt.title(title)
            if len(img0.shape) == 2:
                plt.imshow(img0, cmap='gray')
            else:
                plt.imshow(img0)
            plt.subplot2grid((2, cols), (1, m), colspan=1, rowspan=1)
            plt.plot(cdf_normalized, color='b')
            plt.hist(img.flatten(), 256, [0, rng], color='0.30')
            plt.xlim([0, rng])
            plt.legend(('cdf', 'histogram'), loc='upper left')

            i = i + 1
            if (i > rows):
                i = 0
                j = j + 1
            m = m + 1
        plt.tight_layout()
        if save_image_filename:
            os.makedirs(os.path.dirname(save_image_filename), exist_ok=True)
            plt.savefig(save_image_filename)  # save the figure to file
        if autoclose == -1:
            plt.close("all")
        else:
            if autoclose > 0:
                try:
                    plt.show(block=False)
                    plt.pause(autoclose)  # 3 seconds, I use 1 usually
                except:
                    print("Interrupted while waiting.")
                plt.close("all")
            else:
                plt.show()
            plt.close()

    @staticmethod
    def plot_3dsurface(img, resize=0.15, save_image_filename="", autoclose=0):
        """Plot a surface in 3D

        :Parameters: image, resize=0.15
        """
        img = ims.Image.resize(img, resize)
        xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        fig = plt.figure()
        window_title="3DPlot"
        try:
            plt.gcf().canvas.set_window_title(window_title)
        except:
            plt.gcf().canvas.setWindowTitle(window_title)

        ax = fig.add_subplot(1, 2, 1, projection='3d')

        ax.plot_surface(xx, yy, img, rstride=1, cstride=1, cmap=plt.cm.gray,
                        linewidth=0)
        if save_image_filename:
            os.makedirs(os.path.dirname(save_image_filename), exist_ok=True)
            plt.savefig(save_image_filename)  # save the figure to file
        if autoclose == -1:
            plt.close("all")
        else:
            if autoclose > 0:
                try:
                    plt.show(block=False)
                    plt.pause(autoclose)  # 3 seconds, I use 1 usually
                except:
                    print("Interrupted while waiting.")
                plt.close("all")
            else:
                plt.show()
            plt.close()


