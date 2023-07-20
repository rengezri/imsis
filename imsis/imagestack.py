#!/usr/bin/env python

"""
This file is part of IMSIS
Licensed under the MIT license:
http://www.opensource.org/licenses/MIT-license

This module contains methods for image processing of image stacks
"""

import os
import sys

import cv2 as cv
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.misc
from matplotlib import pyplot as plt
import numpy.random as random
import imsis as ims
import numpy.random as random


class ImageStack:
    @staticmethod
    def to_video(image_list, file_out=r'video.avi', framerate=15):
        """image list (in memory) to video

        :Parameters: image_list
        :Returns: video
        """

        os.makedirs(os.path.dirname(file_out), exist_ok=True)

        frame = image_list[0]
        height, width = frame.shape[0], frame.shape[1]

        video = cv.VideoWriter(file_out, 0, framerate, (width, height))

        for image in image_list:
            image = ims.Image.Convert.toRGB(image)
            video.write(image)
        cv.destroyAllWindows()
        video.release()

    @staticmethod
    def play(image_list, framerate=15, verbose=True, window_name="Imagestack"):
        """view image list (in memory)
        """
        wk = int(1000 / framerate)
        if wk < 1:
            wk = 1
        if verbose == True:
            print("imagestack play: delay per frame in ms {}".format(wk))

        cv.imshow(window_name, image_list[0])
        cv.moveWindow(window_name, 0, 0)

        for image in image_list:
            image = ims.Image.Convert.toRGB(image)
            cv.imshow(window_name, image)

            k = cv.waitKey(wk)
            if k == 27:  # Esc key to stop
                break

        cv.destroyWindow(window_name)

    @staticmethod
    def integrate_images(framelist):
        """compute the integration of n frames
        the last frames weigh heavier
        :Parameters: list of frames
        :Returns: image
        """
        frames = len(framelist)
        out = framelist[0]
        for i in range(1, frames):
            out = cv.addWeighted(out, (1 - 1 / frames), framelist[i], (1 / frames), -1)
        return out

    @staticmethod
    def average_images(framelist):
        """compute the average of n frames
        all frames have the same weight
        :Parameters: list of frames
        :Returns: image
        """
        if (len(framelist)) > 0:
            if (framelist[0].dtype == np.uint8):
                out = np.average(framelist, axis=0).astype(dtype=np.uint8)
            else:
                out = np.average(framelist, axis=0).astype(dtype=np.uint16)
        else:
            print("Error: Insufficient frames in list.")
        return out

    @staticmethod
    def median_images(framelist):
        """compute the median of n frames
        the frames with the most frequent features weight more
        :Parameters: list of frames
        :Returns: image
        """
        out = np.median(framelist, axis=0).astype(dtype=np.uint8)
        return out

    @staticmethod
    def reverse(framelist):
        """reverse frames
        :Parameters: list of frames
        :Returns: list of frames
        """
        return framelist[::-1]

    @staticmethod
    def load(path_in):
        """load image stack

        :Parameters: path
        :Returns: image_list
        """
        filelist = []
        for filename in sorted(os.listdir(path_in)):
            if filename.endswith(('.tiff', '.png', '.tif', '.bmp', '.jpg', '.TIFF', '.PNG', '.TIF', '.BMP', '.JPG')):
                fn = os.path.join(path_in, filename)
                filelist.append(fn)
        image_list = []
        print(filelist)
        for fn in filelist:
            img = ims.Image.load(fn)

            image_list.append(img)
        return image_list

    @staticmethod
    def save(image_list, path_out):
        """save image stack
        :Parameters: image_list, path
        """
        i = 0
        for im3 in image_list:
            cv.imwrite(path_out + "im{:02d}.png".format(i), im3)
            i = i + 1

    @staticmethod
    def transition_dissolve(image0, image1, duration=5 * 15):
        """
        transition dissolve within n frames

        :Parameters: image0, image1
        :Returns: duration in frames
        """
        step = 1 / duration
        alphas = [1 - (step * x) for x in range(0, duration)]
        s = []
        gamma = 1
        for x in range(duration):
            a = image0
            b = image1
            alpha = alphas[x]
            beta = 1 - alpha
            dissolved = cv.addWeighted(a, alpha, b, beta, gamma)
            s.append(dissolved)
        return s

    @staticmethod
    def transition_fadeinout(image0, duration=5 * 15, fadein=True):
        """
        transition fade or out within n frames
        :Parameters: image0, image1
        :Returns: duration in frames, fade in or fade out
        """
        mask = np.zeros(image0.shape, dtype='uint8')
        step = 1 / duration
        alphas = [x * step for x in range(1, duration + 1)]
        s = []
        gamma = 1

        for x in range(duration):
            if fadein:
                alpha = alphas[x]
                beta = alphas[-x]
            else:
                alpha = alphas[-x]
                beta = alphas[x]
            combined = cv.addWeighted(image0, alpha, mask.copy(), beta, gamma)
            s.append(combined)
        return s

    @staticmethod
    def transition_wipe(image0, image1, duration=5 * 15, reverse=False, horizontal=True, showline=True):
        """
        transition wipe, reverse or not, horizontal or vertical
        :Parameters: image0, image1
        :Returns: duration in frames, reverse, horizontal
        """
        s = []

        if horizontal == True:
            w = image0.shape[1]
            step = (w / duration)
        else:
            w = image0.shape[0]
            step = (w / duration)

        x = 0

        frame0 = image0.copy()
        frame1 = image1.copy()
        for i in range(0, duration):
            image0 = frame1.copy()
            image1 = frame0.copy()
            if reverse == False:
                x = int((i + 1) * step)
            else:
                x = w - int((i + 1) * step)
            cval = x
            if (cval < 0):
                cval = 0
            if (cval >= w):
                cval = w - 1
            if horizontal == True:
                image0[0:image0.shape[0], cval:image0.shape[1]] = image1[0:image1.shape[0], cval:image1.shape[1]]
                if showline == True:
                    cv.line(image0, (cval, 0), (cval, image0.shape[1]), (255, 255, 255), 1)
            else:
                image0[cval:image0.shape[0], 0:image0.shape[1]] = image1[cval:image1.shape[0],
                                                                  0:image1.shape[1]]  # note on np crop: y0,y1,x0,x1
                if showline == True:
                    cv.line(image0, (0, cval), (image0.shape[1], cval), (255, 255, 255), 1)

            # cv.line(image0, (cval, 0), (cval, image0.shape[1]), (255, 255, 255), 2)
            s.append(image0)
        return s

    @staticmethod
    def scroll_old(image0, zoomfactor=2, duration=5 * 15, horizontal=True, reverse=False):
        """
        Scroll left - DEPRECIATED
        :Parameters: image0, zoomfactor
        :Returns: frames
        """
        s = []

        if horizontal == True:
            w = image0.shape[1]
            step = (w / duration)
        else:
            h = image0.shape[0]
            step = (h / duration)

        x = 0.5
        y = 0.5
        frame0 = image0.copy()
        for i in range(0, duration):
            image0 = frame0.copy()
            if horizontal == True:
                if reverse == False:
                    x = (int((i + 1) * step)) / frame0.shape[0]
                else:
                    x = (w - int((i + 1) * step)) / frame0.shape[0]
            else:
                if reverse == False:
                    y = (int((i + 1) * step)) / frame0.shape[1]
                else:
                    y = (h - int((i + 1) * step)) / frame0.shape[1]
            image1 = ims.Image.zoom(image0, factor=zoomfactor, cx=x, cy=y)
            s.append(image1)
            # Image.plot(image1)
        return s

    @staticmethod
    def scroll(image0, image1, duration=5 * 15, horizontal=True, reverse=False):
        """
        Scroll left/right/top/down
        :Parameters: image0, zoomfactor
        :Returns: frames
        """

        s = []
        if horizontal == True:
            if reverse == True:
                imagen = ims.Image.Tools.patches2image([image1, image0], cols=2, overlap_px=0, whitebackground=False)
            else:
                imagen = ims.Image.Tools.patches2image([image0, image1], cols=2, overlap_px=0, whitebackground=False)
            w = image0.shape[1]
            x_speed = (w / duration)
            y_speed = 0
            h = image0.shape[0]
        else:
            if reverse == True:
                imagen = ims.Image.Tools.patches2image([image1, image0], cols=1, overlap_px=0, whitebackground=False)
            else:
                imagen = ims.Image.Tools.patches2image([image0, image1], cols=1, overlap_px=0, whitebackground=False)

            h = image0.shape[0]
            y_speed = (h / duration)
            x_speed = 0
            w = image0.shape[1]

        for i in range(0, duration):
            x = int(max(0, min(w, 0 + round(x_speed * i))))
            y = int(max(0, min(h, 0 + round(y_speed * i))))
            imageout = imagen[y: y + h, x: x + w]
            s.append(imageout)
        return s

    @staticmethod
    def create_dummy_imagestack(image0, slices=10, add_noise=0.2, add_translations=0.2, add_rotations=360):
        """
        create dummy slices for an image stack with noise or translations
        noise sigma 0-1, 0=no noise
        translations 0-1, 0= no translation, 1 is full image max
        rotations 0-360, 0 = no rotations

        :Parameters: image, slices, add_noise,add_translations
        :Returns: frames
        """
        w = image0.shape[1]
        h = image0.shape[0]
        s = []
        for i in range(0, slices):
            image1 = image0.copy()
            if add_noise > 0:
                image1 = ims.Image.Process.poisson_noise(image1, prob=add_noise)
            if add_translations > 0:
                shiftx = int(-w * add_translations / 2 + random.randint(0, int(w * add_translations)))
                shifty = int(-h * add_translations / 2 + random.randint(0, int(h * add_translations)))
                image1 = ims.Image.Transform.translate(image1, shiftx, shifty)
            if add_rotations > 0:
                alpha = random.randint(0, add_rotations)
                image1 = ims.Image.Transform.rotate(image1, alpha)

            s.append(image1)
        return s

    @staticmethod
    def align_images(framelist, update_reference_image=True, high_precision=False):
        """
        Correct image shifts for a list of images in memory
        Images should be of the same size

        update_reference_image True: means the last aligned image acts as a reference. False the first image acts as a reference.
        high precision uses a nonmaximum suppression algorithm: this is more precise but a lot slower

        :Parameters: image list
        :Returns: image list, correctiondata list(X,Y,Score)
        """
        framelist2 = []
        correctiondatalist = []
        frames = len(framelist)
        frame0 = framelist[0]
        framelist2.append(frame0)
        correctiondatalist.append([0, 0, 1])
        for i in range(1, frames):
            frame1 = framelist[i]
            if high_precision == True:
                frame2, sx, sy, score = ims.Analyze.ImageAlignment.NONMAX(frame0, frame1)
            else:
                frame2, sx, sy, score = ims.Analyze.ImageAlignment.NCC(frame0, frame1)
            framelist2.append(frame2)
            correctiondatalist.append([sx, sy, score])
            if update_reference_image == True:
                frame0 = frame2.copy()
        return framelist2, correctiondatalist

    @staticmethod
    def align_images_reapply(framelist, correctiondatalist):
        """
        Correct image shifts for a list of images in memory using a correctionlist in memory
        the correction list can be loaded from disk.

        :Parameters: image list, correctiondata list(X,Y,Score)
        :Returns: image list
        """

        framelistnew = []
        i = 0
        for im in framelist:
            x, y, score = correctiondatalist[i]
            im2 = ims.Image.Transform.translate(im, x, y)
            framelistnew.append(im2)
            i = i + 1
        return framelistnew
