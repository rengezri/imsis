#!/usr/bin/env python

"""
This file is part of IMSIS
Licensed under the MIT license:
http://www.opensource.org/licenses/MIT-license

This module contains methods to analyze images
"""

import math

import cv2 as cv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import imsis as ims
from scipy import ndimage
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import scipy
import sys

import scipy.stats as stats
from scipy import fftpack

colors = [
    [255, 255, 64],
    [255, 64, 255],
    [192, 255, 64],
    [192, 64, 255],
    [255, 192, 64],
    [255, 64, 192],
    [192, 192, 64],
    [192, 64, 192],
    [128, 255, 64],
    [128, 64, 255],
    [255, 128, 64],
    [255, 64, 128],
    [128, 192, 64],
    [128, 64, 192],
    [192, 128, 64],
    [192, 64, 128],
    [255, 64, 64],
    [128, 128, 64],
    [128, 64, 128],
    [192, 64, 64],
    [128, 64, 64],
]


class Analyze(object):

    @staticmethod
    def __findlocalExtrema(y):
        # find local maxima of array, including centers of repeating elements
        peakind2 = []
        peakind3 = []
        peakind4 = []  # only valleys of peakind3.
        peakdist = []
        # first filter signs
        lastsign = -1
        for i in range(1, len(y)):
            # dt = (peakdist[i])
            # if (dt > mn + (std * sigma)):
            signchange = y[i] - y[i - 1]
            if signchange > 0:
                signchange = 1
            else:
                signchange = -1
            if (signchange != lastsign):
                peakind2.append(i - 1)
                lastsign = signchange

        for i in range(1, len(peakind2)):
            dt = y[peakind2[i]] - y[peakind2[i - 1]]
            dt2 = math.fabs(dt)
            peakdist.append(dt2)

        # verify length if length is larger than std*sigma than add and verify sign change
        mn = np.median(peakdist)
        std = np.std(peakdist)
        sigma = 2

        for i in range(0, len(peakind2) - 1):
            dt = (peakdist[i])
            if (dt > mn + std):
                peakind3.append(peakind2[i])

        for i in range(0, len(peakind3)):
            a = y[peakind3[i]]
            if (a < 0):
                # print(peakind3[i])
                peakind4.append(peakind3[i])
        return peakind4

    @staticmethod
    def find_edge(img, center, width, height, angle=0, pixelsize=1, derivative=1, invert=False, plotresult=True,
                  verbose=True, autoclose=0):
        """Find edge

         Derivative 1 = first derivative, Derivative 2 = second derivative Derivative 3 = argmax

        :Parameters: image, center, width, height, angle=0, pixelsize=1, derivative=1, invert=False, plotresult=True
        :Returns: image, xpos, ypos
        """
        # different defaults
        twidth = width
        width = height
        height = twidth

        if (invert == True):
            img0 = ims.Image.Adjust.invert(img)
        else:
            img0 = img

        xn, yn = Analyze.get_lineprofile(img0, center, width, height, angle, pixelsize, False)
        y1 = np.gradient(yn)
        y2 = np.gradient(y1)

        if (derivative == 3):
            peakind2 = [np.argmax(yn)]
            # print(peakind2)
        else:
            # use 2nd derivative unless set to 1
            if (derivative == 1):
                peakind2 = Analyze.__findlocalExtrema(y1)
            else:
                peakind2 = Analyze.__findlocalExtrema(y2)
        widthdiv2 = (width / 2)
        heightdiv2 = (height / 2)

        try:
            rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        except:
            rgb = img

        rgb = Analyze.__rotatedrectangle(rgb, (center[0] - int(width / 2), center[1] - int(height / 2)),
                                         (center[0] + int(width / 2), center[1] + int(height / 2)), angle)

        yf = int(peakind2[0])
        x0t = int((center[0] - ((width / 2) - yf) * np.cos(angle * np.pi / 180.)))
        y0t = int((center[1] - ((width / 2) - yf) * np.sin(
            angle * np.pi / 180.)))  # note the length is relevant, therefore use width also for y

        xpos = x0t
        ypos = y0t
        if (verbose == True):
            print('find edge: position found {0},{1}'.format(xpos, ypos))

        rgb = Analyze.__rotatedrectangle(rgb, (xpos - 2, ypos - 2), (xpos + 2, ypos + 2), 0)

        if (plotresult == True):
            plt.figure(figsize=(8, 8))
            plt.gcf().canvas.set_window_title('Find Edges - Image - 1st Derivative, 2nd Derivative')
            gridspec.GridSpec(4, 1)
            plt.subplot2grid((4, 1), (0, 0), colspan=1, rowspan=2)
            st1 = "Edge position=({},{})".format(xpos, ypos)
            plt.title(st1)
            plt.imshow(rgb)
            plt.subplot2grid((4, 1), (2, 0), colspan=1, rowspan=1)
            plt.plot(xn, yn)
            plt.axvline(x=yf,color='g', linestyle='--')

            # Show the major grid lines with dark grey lines
            plt.grid(b=True, which='major', color='#666666', linestyle='-')
            # Show the minor grid lines with very faint and almost transparent grey lines
            plt.minorticks_on()
            plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            #  plt.grid(color='green', linestyle='--', linewidth=0.5)

            plt.subplot2grid((4, 1), (3, 0), colspan=1, rowspan=1)
            plt.plot(xn, y2)
            plt.axvline(x=yf,color='g', linestyle='--')

            # Show the major grid lines with dark grey lines
            plt.grid(b=True, which='major', color='#666666', linestyle='-')
            # Show the minor grid lines with very faint and almost transparent grey lines
            plt.minorticks_on()
            plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            #  plt.grid(color='green', linestyle='--', linewidth=0.5)

            plt.tight_layout()
            if autoclose > 0:
                try:
                    plt.show(block=False)
                    plt.pause(autoclose)  # 3 seconds, I use 1 usually
                except:
                    print("interrupted while waiting.")
                plt.close("all")
            else:
                plt.show()

        return rgb, xpos, ypos

    # returns spot distance in pixels
    @staticmethod
    def find_brightest_spot(img, pixelsize=1, smooth=3, thresh=25, applythresh=True, verbose=True):
        """Find brightest spot

        :Parameters: image, pixelsize=1, smooth=3, thresh=25, applythresh=True
        :Returns: image, xpos, ypos
        """

        def getbrightestspot(img):
            (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(img)
            return maxLoc[0], maxLoc[1]

        if (applythresh == True):
            gray = ims.Image.Adjust.thresholdrange(img, 3, thresh)
        gray = cv.GaussianBlur(img, (smooth, smooth), 0)
        h = int(img.shape[0] / 2.0)
        w = int(img.shape[1] / 2.0)

        # using brightest spot from the top and bottom to find the correct center
        x0, y0 = getbrightestspot(gray)
        gray2 = cv.flip(gray, -1)
        x1, y1 = getbrightestspot(gray2)
        x1 = (w * 2) - x1
        y1 = (h * 2) - y1

        xf = x0 + int((x1 - x0) * 0.5)
        yf = y0 + int((y1 - y0) * 0.5)

        dx = w - xf
        dy = (yf - h) * -1
        dxs = dx * pixelsize
        dys = dy * pixelsize

        linesizediv2 = int(gray.shape[0] / 7 / 2)
        try:
            rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        except:
            rgb = gray

        # cv.rectangle(rgb,(x0,y0),(x1,y1),color=(0,255,0),thickness=2)
        cv.line(rgb, (xf - linesizediv2, yf), (xf + linesizediv2, yf), color=(0, 255, 0), thickness=2)
        cv.line(rgb, (xf, yf - linesizediv2), (xf, yf + linesizediv2), color=(0, 255, 0), thickness=2)
        if verbose == True:
            print("find_brightest_spot: {}".format((dxs, dys)))
        return rgb, dxs, dys

    @staticmethod
    def __imagetolineprofile(img, center, angle, width, height, pixelsize):
        patch = Analyze.__cropsubimage(img, (int(center[0]), int(center[1])), angle, int(width), int(height))
        rows = patch.shape[1]
        cols = patch.shape[0]
        arr = np.array(patch)
        y = np.zeros(rows)
        x = np.zeros(rows)
        # averaged line profile
        for i in range(0, cols):
            for j in range(0, rows):
                x[j] = j
                y[j] = y[j] + arr[i, j]
        for i in range(0, rows):
            y[i] = y[i] / cols
        return x, y

    @staticmethod
    def __cropsubimage(image, center, theta, width, height):
        theta *= np.pi / 180  # convert to rad
        v_x = (math.cos(theta), math.sin(theta))
        v_y = (-math.sin(theta), math.cos(theta))
        s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
        s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)
        mapping = np.array([[v_x[0], v_y[0], s_x],
                            [v_x[1], v_y[1], s_y]])
        return cv.warpAffine(image, mapping, (width, height), flags=cv.WARP_INVERSE_MAP, borderMode=cv.BORDER_REPLICATE)

    @staticmethod
    def get_lineprofile(img, center, width, height, angle=0, pixelsize=1, plotresult=True, autoclose=0):
        """Get line profile by cutting out an image

        :Parameters: image, center, width, height, angle=0, pixelsize=1, plotresult=True
        :Returns: xpos, ypos
        """
        try:
            rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        except:
            rgb = img

        rgb = Analyze.__rotatedrectangle(rgb, (int(center[0]) - int(width / 2), int(center[1]) - int(height / 2)),
                                         (int(center[0]) + int(width / 2), int(center[1]) + int(height / 2)), angle)

        #    rgb = cv.rectangle(rgb, (center[0]-int(width/2),center[1]-int(height/2)), (center[0]+int(width/2),center[1]+int(height/2)), (0, 255, 0), 3)
        x, y = Analyze.__imagetolineprofile(img, center, angle, width, height, pixelsize)

        for i in range(0, len(x)):
            x[i] = x[i] * pixelsize

        if (plotresult == True):
            plt.figure(figsize=(8, 8))
            plt.gcf().canvas.set_window_title('Line Profile')
            gridspec.GridSpec(3, 1)
            plt.subplot2grid((3, 1), (0, 0), colspan=1, rowspan=2)
            plt.imshow(rgb)
            plt.subplot2grid((3, 1), (2, 0), colspan=1, rowspan=1)
            plt.plot(x, y)
            # Show the major grid lines with dark grey lines
            plt.grid(b=True, which='major', color='#666666', linestyle='-')
            # Show the minor grid lines with very faint and almost transparent grey lines
            plt.minorticks_on()
            plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            #  plt.grid(color='green', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            if autoclose > 0:
                try:
                    plt.show(block=False)
                    plt.pause(autoclose)  # 3 seconds, I use 1 usually
                except:
                    print("interrupted while waiting.")
                plt.close("all")
            else:
                plt.show()
        return x, y

    @staticmethod
    def __rotatedrectangle(img, p1, p2, angle):

        def rotate2d(pts, cnt, ang=np.pi / 180):
            pts = np.dot(pts - cnt, np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])) + cnt
            for i in range(0, len(pts)):
                pts[i] = (int(pts[i][0]), int(pts[i][1]))
            pts2 = pts.astype(int)
            return pts2

        pts = np.array([[p1[0], p1[1]], [p2[0], p1[1]], [p2[0], p2[1]], [p1[0], p2[1]]], np.int32)
        # print(pts)
        center = ((p1[0] + int(p2[0] - p1[0]) / 2.0), (p1[1] + int(p2[1] - p1[1]) / 2.0))
        # print(center)

        pts = rotate2d(pts, center, np.pi / 180 * angle)
        # print(pts)
        cv.polylines(img, [pts], True, (0, 255, 0), thickness=2)
        return img

    @staticmethod
    def measure_lines(img, lines, linewidth=50, pixelsize=0, derivative=1, invert=False, roundangles=1, fontsize=40,
                      verbose=True, autoclose=0):
        """Weasure the width of a line more accurately by applying edgefinders at both ends returns image and line measurements
        pixelsize=0 expresses measurement in pixels anything else results in a metric representation
        roundangles = 1 is no rounding, roundangles = 5 round to 5 degree angles

        :Parameters: image, lines, linewidth=50, pixelsize=1, derivative=1, invert=False, plotresult=True
        :Returns: image, xpos, ypos
        """

        try:
            rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        except:
            rgb = img

        measurementlist = []
        for line in lines:
            print(line)  # [shapenr][(x0,y0),(x1,y1)]

            position, length, angle = Analyze.vector_position_length_direction(line)
            center = (int(position[0]), int(position[1]))
            # angle = int(angle) - int(angle) % 5  # round to units of 5
            angle = int(round(angle / roundangles) * roundangles)  # round to units of 15
            length = int(length)  # round to integer

            ang0 = angle
            ang1 = angle + 180

            img4, x0, y0 = Analyze.find_edge(img, center, linewidth, length, ang0, pixelsize, derivative=derivative,
                                             invert=invert, plotresult=False, verbose=False)
            img4, x1, y1 = Analyze.find_edge(img, center, linewidth, length, ang1, pixelsize, derivative=derivative,
                                             invert=invert, plotresult=False, verbose=False)
            if verbose == True:
                print("Measure Line: {},{},{},{}".format(x0, y0, x1, y1))
            # widthout = np.abs(y0 - y1)*pixelsize

            rgb, length2 = Analyze.add_singleline_measurement(rgb, x0, y0, x1, y1, pixelsize, linethickness=1,
                                                              fontsize=fontsize,
                                                              tiltcorrection=0, verbose=verbose)
            measurementlist.append(length2)

        if verbose == True:
            print("measure lines: list of lines: {}".format(measurementlist))
            ims.View.plot(rgb, 'Measure Lines', window_title='Measure Lines', autoclose=autoclose)

        return rgb, measurementlist

    @staticmethod
    def measure_linewidth(img, center, width, height, angle, pixelsize=0, derivative=1, linethickness=1, invert=False,
                          plotresult=True,
                          plotboundingbox=True, verbose=True, autoclose=0):
        """"Measure the width of a line more accurately by applying edgefinders at both ends. can measure lines under different angles

        :Parameters: image, lines, linewidth=50, pixelsize=1, derivative=1, invert=False, plotresult=True
        :Returns: image, length
        """
        ang0 = angle
        ang1 = angle + 180

        img4, x0, y0 = Analyze.find_edge(img, center, width, height, ang0, pixelsize, derivative=derivative,
                                         invert=invert, plotresult=False, verbose=False)
        img4, x1, y1 = Analyze.find_edge(img, center, width, height, ang1, pixelsize, derivative=derivative,
                                         invert=invert, plotresult=False, verbose=False)

        if verbose == True:
            print("Measure Line: {},{},{},{}".format(x0, y0, x1, y1))

        xpos0 = x0
        xpos1 = x1
        ypos0 = y0
        ypos1 = y1
        widthout = np.abs(y0 - y1)
        try:
            rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        except:
            rgb = img

        rgb, length2 = ims.Analyze.add_singleline_measurement(rgb, x0, y0, x1, y1, pixelsize, linethickness=1,
                                                              fontsize=40,
                                                              tiltcorrection=0)
        ims.View.plot(rgb, 'Get Linewidth', window_title='Measure Linewidth', autoclose=autoclose)
        if verbose == True:
            print("get_linewidth: {}".format(length2))
        return img4, length2

    # Measure spheres and return image
    @staticmethod
    def measure_spheres(img, threshimg, pixelsize=1, areamin=5, areamax=100000, addtext=True, verbose=True):
        """"get size of spherical shapes in an image

        :Parameters: image, threshimg, pixelsize=1,areamin=200,areamax=10000
        :Returns: image, sphere_list
        """

        try:
            rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        except:
            rgb = img

        # cv version check
        (cvmajor, cvminor, _) = cv.__version__.split(".")
        if (int(cvmajor) < 4):
            im2, contours, hierarchy = cv.findContours(threshimg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        else:
            contours, hierarchy = cv.findContours(threshimg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        font = cv.FONT_HERSHEY_SIMPLEX

        mlist = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            if (area > areamin and area < areamax):
                ellipse = cv.fitEllipse(cnt)
                (position, size, angle) = ellipse
                # print(position, size)
                rgb = cv.ellipse(rgb, ellipse, (0, 255, 0), 2)

                unit = 1e-3
                unittxt = 'mm'
                if ((size[0] * pixelsize) < 1e-6):
                    unit = 1e9  # nm
                    unittxt = 'nm'
                else:
                    unit = 1e6  # um
                    unittxt = u'\u03bc' + 'm'

                text = '{0:.2f}'.format(size[0] * pixelsize * unit) + unittxt + '\n'
                text = text + '{0:.2f}'.format(size[1] * pixelsize * unit) + unittxt + ''
                postext = (int(position[0]), int(position[1] + size[1] * 0.5 * 1.2))
                if addtext == True:
                    rgb = Analyze.add_text(rgb, postext[0], postext[1], text, fontsize=20, aligntocenter=True)
                mlist.append(
                    [[position[0] * pixelsize, position[1] * pixelsize], [size[0] * pixelsize, size[1] * pixelsize],
                     angle])
        if verbose == True:
            print("measure spheres, spheres found: {}".format(mlist))

        return rgb, mlist

    @staticmethod
    def create_lineiterator(img, P1, P2):
        """"line profile of an image from point1 to point2

        :Parameters: image, point1, point2
        :Returns: float_list
        """

        # https://stackoverflow.com/users/3098020/mohikhsan, modified
        # lineiterator opencv python is broken
        # define local variables for readability
        imageH = img.shape[0]
        imageW = img.shape[1]
        P1X = P1[0]
        P1Y = P1[1]
        P2X = P2[0]
        P2Y = P2[1]

        # difference and absolute difference between points
        # used to calculate slope and relative location between points
        dX = P2X - P1X
        dY = P2Y - P1Y
        dXa = np.abs(dX)
        dYa = np.abs(dY)

        # predefine numpy array for output based on distance between points
        itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
        itbuffer.fill(np.nan)

        # Obtain coordinates along the line using a form of Bresenham's algorithm
        negY = P1Y > P2Y
        negX = P1X > P2X
        if P1X == P2X:  # vertical line segment
            itbuffer[:, 0] = P1X
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
        elif P1Y == P2Y:  # horizontal line segment
            itbuffer[:, 1] = P1Y
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
        else:  # diagonal line segment
            steepSlope = dYa > dXa
            if steepSlope:
                slope = dX / dY
                if negY:
                    itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
                else:
                    itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
                itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(np.int) + P1X
            else:
                slope = dY / dX
                if negX:
                    itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
                else:
                    itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
                itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(np.int) + P1Y

        # Remove points outside of image
        colX = itbuffer[:, 0]
        colY = itbuffer[:, 1]
        itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

        # Get intensities from img ndarray
        itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]
        zi = itbuffer[:, 2]
        return zi

    @staticmethod
    def vector_position_length_direction(vector):
        """"return the position, length and direction of a vector
        position (center_x, center_y)
        direction (angle in degrees)

        :Parameters: vector
        :Returns: length, direction
        """
        x0 = vector[0][0]
        y0 = vector[0][1]
        x1 = vector[1][0]
        y1 = vector[1][1]
        direction = (np.arctan2((y1 - y0), (x1 - x0))) * 180.0 / np.pi
        length = np.power(np.power(x1 - x0, 2.0) + np.power(y1 - y0, 2.0), 0.5)

        x0, y0, x1, y1 = vector[0][0], vector[0][1], vector[1][0], vector[1][1]
        cx = x0 + int((x1 - x0) * 0.5)
        cy = y0 + int((y1 - y0) * 0.5)
        position = (cx, cy)
        return position, length, direction

    @staticmethod
    def rectangle_pixels(shape):
        """returns rectangle shape into (width,height,centerx,centery)

        :Parameters: shape
        :Returns: width, height, centerx, centery
        """
        x0 = shape[0][0]
        y0 = shape[0][1]
        x1 = shape[1][0]
        y1 = shape[1][1]
        width = np.abs(x0 - x1)
        height = np.abs(y0 - y1)
        centerx = int(x0 + width * 0.5)
        centery = int(y0 + height * 0.5)
        return width, height, centerx, centery

    @staticmethod
    def rectangle_pixels_to_percentage(im, rectangle, verbose=False):
        """convert a selected rectangle in pixels to a position and size (between 0 and 1 with 0.5 being the center)

        :Parameters: image rectangle
        :Returns: widthperc, heightperc, centerxperc, centeryperc
        """

        # shape x0,y0,x1,y1
        x0 = rectangle[0][0]
        y0 = rectangle[0][1]
        x1 = rectangle[1][0]
        y1 = rectangle[1][1]

        imwidth = im.data.shape[1]
        imheight = im.data.shape[0]
        # print(imwidth, imheight)

        width = np.abs(x0 - x1)
        height = np.abs(y0 - y1)
        centerx = int(x0 + width * 0.5)
        centery = int(y0 + height * 0.5)
        # print(width, height, centerx, centery)
        widthperc = width / imwidth
        heightperc = height / imheight
        centerxperc = centerx / imwidth
        centeryperc = centery / imheight
        if verbose == True:
            print(widthperc, heightperc, centerxperc, centeryperc)
        return widthperc, heightperc, centerxperc, centeryperc

    @staticmethod
    def __rectanglepositiontopixels(imwidth, imheight, t_center, t_size):
        t_center_pixels = [ims.Measure.multipleof2(imwidth * t_center[0]),
                           ims.Measure.multipleof2(imheight * t_center[1])]
        t_size_pixels = [ims.Measure.multipleof2(imwidth * t_size[0]), ims.Measure.multipleof2(imheight * t_size[1])]
        return t_center_pixels, t_size_pixels

    @staticmethod
    def find_contour_center(img, verbose=True):
        """Find spot centers using contours

        :Parameters: image
        :Returns: image, centerx, centery
        """
        try:
            rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        except:
            rgb = img

        # cv version check
        (cvmajor, cvminor, _) = cv.__version__.split(".")
        if (int(cvmajor) < 4):
            im2, contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        else:
            contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        mlist = []
        cX = 0
        cY = 0
        for cnt in contours:
            area = cv.contourArea(cnt)
            if (area > 200 and area < 10000):
                M = cv.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # draw the contour and center of the shape on the image
                cv.drawContours(rgb, [cnt], -1, (0, 255, 0), 2)
                cv.circle(rgb, (cX, cY), 7, (0, 255, 0), -1)
                # print(cX, cY)
        if verbose == True:
            print("find contour center: {}".format((cX, cY)))
        return rgb, cX, cY

    @staticmethod
    def find_image_center_of_mass(im, verbose=True):
        """determine image center-of-mass, this can be used to find a single bright spot on a dark image

        :Parameters: image
        :Returns: image, centerx, centery
        """
        try:
            rgb = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
        except:
            rgb = im

        """
        m = np.sum(np.asarray(im), -1) < 255*3
        m = m / np.sum(np.sum(m))

        dx = np.sum(m, 0)  # there is a 0 here instead of the 1
        dy = np.sum(m, 1)  # as np.asarray switches the axes, because
        # in matrices the vertical axis is the main
        # one, while in images the horizontal one is
        # the first
        """
        cm = ndimage.measurements.center_of_mass(im)
        try:
            dx = int(cm[1])
            dy = int(cm[0])
        except:
            print('skip measurement.')
            dx = 0
            dy = 0
        rgb = cv.circle(rgb, (dx, dy), 7, (0, 255, 0), -1)
        if verbose == True:
            print("find image center of mass: {}".format((dx, dy)))
        return rgb, dx, dy

    """
    #failed not picking up the right point

    @staticmethod
    def find_brightest_spot(gray):
        gray = cv.GaussianBlur(gray, (7, 7), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)
        return maxLoc[0],maxLoc[1]
    """

    @staticmethod
    def __pretty_unit(val, baseunit='m'):
        """Give the number an appropriate SI prefix.

        :Parameters: Too big or too small number.
        :Returns: String containing a number between 1 and 1000 and SI prefix.
        """

        def sign(x, value=1):
            """Mathematical signum function.

            :param x: Object of investigation
            :param value: The size of the signum (defaults to 1)
            :returns: Plus or minus value
            """
            return -value if x < 0 else value

        if val == 0:
            return "0" + baseunit

        l = np.floor(np.log10(abs(val)))
        if abs(l) > 24:
            l = sign(l, value=24)

        div, mod = divmod(l, 3 * 1)

        return "%.3g%s" % (val * 10 ** (-l + mod), " kMGTPEZYyzafpnµm"[int(div)]) + baseunit

    @staticmethod
    def add_histogram(img, scale=0.2):
        """Add histogram to an image
        scale 0.2 means 20 perc of the image size

        :Parameters: image, x0=0, y0=0, scale=0.2
        :Returns: image
        """
        img2 = ims.Image.histogram(img)
        widthfull = img.shape[1]
        widthhist = img2.shape[1]
        factor = (widthfull / widthhist) * scale
        img2 = ims.Image.resize(img2, factor)
        img2 = ims.Image.invert(img2)

        out = img
        xoff = img.shape[1] - img2.shape[1]
        out[0:img2.shape[0], xoff:img2.shape[1] + xoff] = img2
        # out = ims.Image.add(img,img2)

        return out

    @staticmethod
    def add_text(img, x0=0, y0=0, text='empty', fontsize=60, aligntocenter=False, outline=True):
        """Add text to an image

        :Parameters: image, x0=0, y0=0, text='empty', fontsize=60, aligntocenter=False
        :Returns: image
        """
        font = ImageFont.truetype("arial.ttf", fontsize)
        cx, cy = font.getsize(text)
        rgb = ims.Image.Convert.toRGB(img)
        pil_im = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_im)
        if (aligntocenter == True):
            x0 = x0 - int(cx * 0.5)
            y0 = y0 - int(cy * 0.5)

        # thin border
        if outline == True:
            shadowcolor = (0, 0, 0)
            draw.text((x0 - 1, y0), text, font=font, fill=shadowcolor)
            draw.text((x0 + 1, y0), text, font=font, fill=shadowcolor)
            draw.text((x0, y0 - 1), text, font=font, fill=shadowcolor)
            draw.text((x0, y0 + 1), text, font=font, fill=shadowcolor)
            draw.text((x0, y0), text, fill=(255, 255, 255), font=font)
        else:
            draw.text((x0, y0), text, fill=(0, 255, 0), font=font)

        img = np.array(pil_im)
        return img

    @staticmethod
    def create_text_label(text, fontsize):
        """Create a text label, by adding a text to an empty bitmap

        :Parameters: text, fontsize
        :Returns: image
        """
        font = ImageFont.truetype("arial.ttf", fontsize)
        textX, textY = font.getsize(text)
        rgb = np.zeros((textY, textX, 3), np.uint8)
        pil_im = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_im)
        draw.text((0, 0), text, 'white', font=font)
        img = np.array(pil_im)
        return img

    @staticmethod
    def add_line_measurements(img, lines, pixelsize=1.0, linethickness=1, fontsize=40, tiltcorrection=0,
                              outline=True, verbose=True):
        """Draw multiple line measurements on an image

        :Parameters: image, x0, y0, x1, y1, pixelsize=1.0, fontsize=40, tiltcorrection=0
        :Returns: image, list_of_lengths
        """
        img1 = img.copy()

        list_of_lengths = []
        for line in lines:
            x0 = line[0][0]
            y0 = line[0][1]
            x1 = line[1][0]
            y1 = line[1][1]

            img1, newlength = Analyze.add_singleline_measurement(img1, x0, y0, x1, y1, pixelsize, linethickness,
                                                                 fontsize, tiltcorrection, outline, verbose)
            list_of_lengths.append(newlength)

        return img1, list_of_lengths

    @staticmethod
    def add_singleline_measurement(img, x0, y0, x1, y1, pixelsize=1.0, linethickness=1, fontsize=40, tiltcorrection=0,
                                   outline=True, verbose=True):
        """Add a lineMeasurement to an image

        :Parameters: image, x0, y0, x1, y1, pixelsize=1.0, fontsize=40, tiltcorrection=0
        :Returns: image, float
        """
        w = img.shape[0]
        h = img.shape[1]
        cx = x0 + int((x1 - x0) * 0.5)
        cy = y0 + int((y1 - y0) * 0.5)

        tiltcorr = np.cos(tiltcorrection * 180.0 / np.pi)
        length = np.sqrt((x0 - x1) ** 2 + ((y0 - y1)) ** 2)
        if outline == True:
            length = length - (linethickness * 2)
        tiplength = 7.0 / length
        length2 = np.sqrt((x0 - x1) ** 2 + ((y0 - y1) * tiltcorr) ** 2)

        img = ims.Image.Convert.toRGB(img)

        if outline == True:
            shadowborder = (0, 0, 0)
            img = cv.arrowedLine(img, (x0 - 1, y0 + 1), (x1 - 1, y1 + 1), shadowborder, thickness=linethickness,
                                 tipLength=tiplength)
            img = cv.arrowedLine(img, (x0 - 1, y0 - 1), (x1 - 1, y1 - 1), shadowborder, thickness=linethickness,
                                 tipLength=tiplength)
            img = cv.arrowedLine(img, (x0 + 1, y0 - 1), (x1 + 1, y1 - 1), shadowborder, thickness=linethickness,
                                 tipLength=tiplength)
            img = cv.arrowedLine(img, (x0 + 1, y0 + 1), (x1 + 1, y1 + 1), shadowborder, thickness=linethickness,
                                 tipLength=tiplength)

            img = cv.arrowedLine(img, (x1 - 1, y1 + 1), (x0 - 1, y0 + 1), shadowborder, thickness=linethickness,
                                 tipLength=tiplength)
            img = cv.arrowedLine(img, (x1 - 1, y1 - 1), (x0 - 1, y0 - 1), shadowborder, thickness=linethickness,
                                 tipLength=tiplength)
            img = cv.arrowedLine(img, (x1 + 1, y1 - 1), (x0 + 1, y0 - 1), shadowborder, thickness=linethickness,
                                 tipLength=tiplength)
            img = cv.arrowedLine(img, (x1 + 1, y1 + 1), (x0 + 1, y0 + 1), shadowborder, thickness=linethickness,
                                 tipLength=tiplength)
            img = cv.arrowedLine(img, (x0, y0), (x1, y1), (255, 255, 255), thickness=linethickness, tipLength=tiplength)
            img = cv.arrowedLine(img, (x1, y1), (x0, y0), (255, 255, 255), thickness=linethickness, tipLength=tiplength)
        else:
            img = cv.arrowedLine(img, (x0, y0), (x1, y1), (0, 255, 0), thickness=linethickness, tipLength=tiplength)
            img = cv.arrowedLine(img, (x1, y1), (x0, y0), (0, 255, 0), thickness=linethickness, tipLength=tiplength)

        if pixelsize == 0:
            text = str(length2)
            print("pixels!!!")
            length_out = length2
        else:
            text = Analyze.__pretty_unit(length2 * pixelsize)
            length_out = length2 * pixelsize

        img = Analyze.add_text(img, cx, cy, text, fontsize, False)
        if verbose == True:
            print('add line measurement: {0}, tiltcorrection: {1}'.format(text, tiltcorr))

        return img, length_out

    @staticmethod
    def add_scalebar(img, pixelsize):
        """Remove the databar and add a scalebar to an image (or add scalebar if no databar exists)
        pixelsize is expressed in meters
        1=1m, 1x1e-3=1mm

        :Parameters: image, pixelsize
        :Returns: image
        """

        def sign(x, value=1):
            """Mathematical signum function.

            :param x: Object of investigation
            :param value: The size of the signum (defaults to 1)
            :returns: Plus or minus value
            """
            return -value if x < 0 else value

        # img = cv.pyrDown(img)
        width, height = tuple(img.shape[1::-1])

        fontsize = int(height * 0.02)

        hfw = (width * pixelsize)  # pixelsize is expressed in meters

        scalebarwidth = int((width * 0.2) / 100) * 100

        val = scalebarwidth
        l = np.floor(np.log10(abs(val)))
        if abs(l) > 24:
            l = sign(l, value=24)
        div, mod = divmod(l, 3 * 1)
        sc = 10 ** (-l + mod)
        # scalebarwidth = float(1/sc)

        print(sc, pixelsize, scalebarwidth)

        text = Analyze.__pretty_unit(scalebarwidth * pixelsize)

        posx = int(width - width / 20)
        posy = int(height - height / 20)

        sizex = int(scalebarwidth)
        sizey = int(sizex / 40)
        sizey = fontsize * 0.5

        posx = posx - int(sizex / 2)
        # print(scalebarwidth, pixelsize)
        # print(posx, posy, sizex, sizey)
        img = cv.rectangle(img, (posx - int(sizex / 2) - 1, posy - int(sizey / 2) - 1),
                           (posx + int(sizex / 2) + 1, posy + int(sizey / 2) + 1),
                           (0, 0, 0), -1)
        img = cv.rectangle(img, (posx - int(sizex / 2), posy - int(sizey / 2)),
                           (posx + int(sizex / 2), posy + int(sizey / 2)),
                           (255, 255, 255), -1)
        # text = str(scalebarwidth) + ' ' + unit

        pil_im = Image.fromarray(img)
        font = ImageFont.truetype("arial.ttf", fontsize)
        draw = ImageDraw.Draw(pil_im)
        textX, textY = font.getsize(text)
        img = Analyze.add_text(img, posx - int(textX / 2), posy - sizey - int(textY), text, fontsize, outline=True)
        # draw.text((posx - int(textX / 2), posy - sizey - int(textY)), text, 'white', font=font)
        # img = np.array(pil_im)

        #    img = cv.cvtColor(np.array(pil_im), cv.COLOR_RGB2BGR)
        return img

    @staticmethod
    def feature_size_distribution(orig, thresh):
        """Determine the distribution of features of different size
        Enter the threshold value for masking. Features will be labeled and counted.

        overlay displays an overlay of the features found on top of the original image.
        out shows the labeled features without overlay
        graph shows the size distribution
        cntsSorted contains the list of contours

        :Parameters: original_image, threshold_image
        :Returns: overlay, out, cntsSorted, (area, colorvalue)
        """

        # img = ims.Image.Convert.toGray(orig)
        # Find all contours on the map
        # _th, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # cv version check
        (cvmajor, cvminor, _) = cv.__version__.split(".")
        if (int(cvmajor) < 4):
            _th, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        else:
            contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        print("Number of contours detected = %d" % len(contours))
        cntsSorted = sorted(contours, key=lambda x: cv.contourArea(x))
        # Fill second level regions on the map
        coln = 0

        minarea = cv.contourArea(cntsSorted[0])
        maxarea = cv.contourArea(cntsSorted[-1])
        mularea = (len(colors) - 1) / (maxarea - minarea)

        img5 = ims.Image.Convert.toRGB(orig)
        img5b = img5.copy()

        # fig = plt.figure(figsize=(12, 6))
        sizedistout = []
        for i in range(len(cntsSorted)):
            area = cv.contourArea(cntsSorted[i])
            colv = area * mularea
            colvi = int(((colv)))

            coln = colors[colvi]
            # coln = (coln + 1) % len(colors)
            img6 = cv.drawContours(img5, cntsSorted, i, coln, -1)
            # out = cv.drawContours(img1,cntsSorted,i,coln,-1)

            col = [coln[0] / 255., coln[1] / 255., coln[2] / 255., 1]
            plt.bar(i, area, color=col)
            plt.xlabel("Contours")
            plt.ylabel("Area")
            sizedistout.append([area, colv])

        out = cv.bitwise_and(img6, img6, mask=thresh)

        overlay = cv.addWeighted(img5b, 0.7, out, 0.3, 0)
        # graph = fig
        # fig.show()
        return overlay, out, cntsSorted, sizedistout

    @staticmethod
    def correct_imageshift_list(framelist):
        """
        Correct image shifts for a list of images in memory
        Images should be of the same size

        :Parameters: image list
        :Returns: image list, correctiondata list(X,Y,Score)
        """
        framelist2 = []
        correctiondatalist = []
        frames = len(framelist)
        frame0 = framelist[0]
        for i in range(1, frames):
            frame1 = framelist[i]
            frame0, sx, sy, score = Analyze.ImageAlignment.NCC(frame0, frame1)
            framelist2.append(frame0)
            correctiondatalist.append([sx, sy, score])
        return framelist2, correctiondatalist

    @staticmethod
    def PSNR(target, ref):
        """Return peaksignal to noise ratio

        :Parameters: image1, image2
        :Returns: float
        """
        target_data = np.array(target, dtype=np.float64)
        ref_data = np.array(ref, dtype=np.float64)

        diff = ref_data - target_data
        # print(diff.shape)
        diff = diff.flatten('C')
        rmse = np.math.sqrt(np.mean(diff ** 2.))
        if (rmse == 0):
            ret = np.inf
        else:
            ret = 20 * np.math.log10(255 / rmse)
        return ret

    @staticmethod
    def rectangle_percentage_to_pixels(img, t_center, t_size):
        imwidth = img.shape[1]
        imheight = img.shape[0]
        t_center_pixels = [ims.Misc.multipleof2(imwidth * t_center[0]),
                           ims.Misc.multipleof2(imheight * t_center[1])]
        t_size_pixels = [ims.Misc.multipleof2(imwidth * t_size[0]), ims.Misc.multipleof2(imheight * t_size[1])]
        return t_center_pixels, t_size_pixels

    @staticmethod
    def compare_image_identical(img0, img1):
        """Compare images, return true if identical

        :Parameters: image0,image1
        :Returns: bool
        """
        result = False
        difference = cv.subtract(img0, img1)
        b, g, r = cv.split(difference)
        if cv.countNonZero(b) == 0 and cv.countNonZero(g) == 0 and cv.countNonZero(r) == 0:
            result = True
        return result

    @staticmethod
    def compare_image_mse(img0, img1):
        """Compare images by mean squared error (mse), return value
        Identical images show an MSE value of 0.
        :Parameters: image0,image1
        :Returns: value
        """
        err = np.sum((img0.astype("float") - img1.astype("float")) ** 2)
        err /= float(img0.shape[0] * img0.shape[1])
        return err

    @staticmethod
    def powerspectrum(image, verbose=True, autoclose=0):
        """ Calculate the powerspectrum of an image
        :Parameters: image
        :Returns: powerspectrum
        """

        def azimuthalAverage(image, center=None):
            """
            Calculate the azimuthally averaged radial profile.

            image - The 2D image
            center - The [x,y] pixel coordinates used as the center. The default is
                     None, which then uses the center of the image (including
                     fracitonal pixels).
            #https://www.astrobetter.com/wiki/tiki-index.php?page=python_radial_profiles
            """
            # Calculate the indices from the image
            y, x = np.indices(image.shape)

            if not center:
                center = np.array([(x.max() - x.min()) / 2.0, (x.max() - x.min()) / 2.0])

            r = np.hypot(x - center[0], y - center[1])

            # Get sorted radii
            ind = np.argsort(r.flat)
            r_sorted = r.flat[ind]
            i_sorted = image.flat[ind]

            # Get the integer part of the radii (bin size = 1)
            r_int = r_sorted.astype(int)

            # Find all pixels that fall within each radial bin.
            deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
            rind = np.where(deltar)[0]  # location of changed radius
            nr = rind[1:] - rind[:-1]  # number of radius bin

            # Cumulative sum to figure out sums for each radius bin
            csim = np.cumsum(i_sorted, dtype=float)
            tbin = csim[rind[1:]] - csim[rind[:-1]]

            radial_prof = tbin / nr

            return radial_prof

        image = ims.Image.Tools.squared(image)
        image = ims.Image.Convert.toGray(image)
        # Take the fourier transform of the image.
        F1 = fftpack.fft2(image)

        # Now shift the quadrants around so that low spatial frequencies are in
        # the center of the 2D fourier transformed image.
        F2 = fftpack.fftshift(F1)

        # Calculate a 2D power spectrum
        psd2D = np.abs(F2) ** 2

        # Calculate the azimuthally averaged 1D power spectrum
        psd1D = azimuthalAverage(psd2D)

        if (verbose == True):
            '''
            # Now plot up both
            plt.figure(1)
            plt.clf()
            plt.imshow(np.log10(image), cmap=plt.cm.Greys)
    
            plt.figure(2)
            plt.clf()
            plt.imshow(np.log10(psd2D))
            '''

            plt.figure(3)
            plt.clf()
            plt.gcf().canvas.set_window_title('Power Spectrum')
            plt.semilogy(psd1D)
            plt.xlabel("Spatial Frequency")
            plt.ylabel("Power Spectrum")

            if autoclose>0:
                try:
                    plt.show(block=False)
                    plt.pause(autoclose)  # 3 seconds, I use 1 usually
                except:
                    print("interrupted while waiting.")
                plt.close("all")
            else:
                plt.show()



        return psd1D

    @staticmethod
    def hough_lines(image, threshold=2, minlinelength=50, maxlinegap=5):
        """ Hough lines detection

        :Parameters: image
        :Returns: image
        """

        newimage = image.copy()
        grayscale = ims.Image.Convert.toGray(newimage)
        # perform edge detection
        edges = cv.Canny(grayscale, 30, 100)
        # detect lines in the image using hough lines technique
        # lines = cv.HoughLinesP(edges, 1, np.pi / 180, 60, np.array([]), 50, 5)
        lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold, minlinelength, maxlinegap)

        h = image.shape[0]
        w = image.shape[1]
        out = np.zeros((h, w, 3), np.uint8)
        # iterate over the output lines and draw them
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # cv.line(edges, (x1, y1), (x2, y2), (255, 0, 0), 3)
        return out, lines

    @staticmethod
    def is_image_empty(img):
        """ Return true if image is empty

        :Parameters: image
        :Returns: bool
        """
        img = ims.Image.Convert.toGray(img)
        result = False
        if cv.countNonZero(img) == 0:
            result = True
        return result

    @staticmethod
    def get_average_color(img):
        """ Return average color in an image
        :Parameters: image
        :Returns: color
        """
        avg_color_per_row = np.average(img, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        return avg_color

    @staticmethod
    def get_dominant_color(img):
        """ Return dominant color in an image
        :Parameters: image
        :Returns: color
        """
        if len(img.shape) == 3:
            colors, count = np.unique(img.reshape(-1, img.shape[-1]), axis=0, return_counts=True)
            out = (colors[count.argmax()]).tolist()
        else:
            colors, count = np.unique(img.reshape(-1, 1), axis=0, return_counts=True)
            out = (colors[count.argmax()]).tolist()
        return out

    class SharpnessDetection:
        # Ref: http: // radjkarl.github.io / imgProcessor / _modules / imgProcessor / measure / sharpness / parameters.html

        def varianceOfLaplacian(self, img):
            """Variance of Laplacian (LAPV) Pech 2000
            """
            lap = cv.Laplacian(img, ddepth=-1)  # cv2.cv.CV_64F)
            stdev = cv.meanStdDev(lap)[1]
            s = stdev[0] ** 2
            return s[0]

        def tenengrad(self, img, ksize=3):
            """Tenegrad (TENG) Krotkov86
            """
            Gx = cv.Sobel(img, ddepth=cv.CV_64F, dx=1, dy=0, ksize=ksize)
            Gy = cv.Sobel(img, ddepth=cv.CV_64F, dx=0, dy=1, ksize=ksize)
            FM = Gx * Gx + Gy * Gy
            mn = cv.mean(FM)[0]
            if np.isnan(mn):
                return np.nanmean(FM)
            return mn

        def tenengradvariance(self, img, ksize=3):
            """Tenegrad variance (TENV) Pech2000
            """
            Gx = cv.Sobel(img, ddepth=cv.CV_64F, dx=1, dy=0, ksize=ksize)
            Gy = cv.Sobel(img, ddepth=cv.CV_64F, dx=0, dy=1, ksize=ksize)
            FM = Gx * Gx + Gy * Gy
            mean, stdev = cv.meanStdDev(img)
            fm = stdev[0] ** 2
            if np.isnan(fm):
                return np.nanmean(FM)
            return fm[0]

        def normalizedGraylevelVariance(self, img):
            """Normalaized Gray Level Variance (GLVN) Santos 97
            """
            mean, stdev = cv.meanStdDev(img)
            s = stdev[0] ** 2 / mean[0]
            return s[0]

        def graylevelVariance(self, img):
            """Gray Level Variance (GLVA) Krotkov86
            """
            mean, stdev = cv.meanStdDev(img)
            s = stdev[0]
            return s[0]

        def modifiedLaplacian(self, img):
            """Modified Laplacian (LAPM) Nayar 89
            """
            M = np.array([-1, 2, -1])
            # G = cv.getGaussianKernel(ksize=3, sigma=-1) #kernel returns [0.25,0.5,0.25] but fails with sepfilter
            G = np.array([1, 2, 1])

            Lx = cv.sepFilter2D(src=img, ddepth=cv.CV_64F, kernelX=M, kernelY=G)
            Ly = cv.sepFilter2D(src=img, ddepth=cv.CV_64F, kernelX=G, kernelY=M)
            FM = np.abs(Lx) + np.abs(Ly)
            return cv.mean(FM)[0]

        def diagonalLaplacian(self, img):
            """Diagonal Laplacian (LAPD) Thelen2009
            """
            M1 = np.array([-1, 2, -1])
            G = np.array([1, 2, 1])
            M2 = np.array([[0, 0, -1], [0, 2, 0], [-1, 0, 0]]) / np.math.sqrt(2)
            M3 = np.array([[-1, 0, 0], [0, 2, 0], [0, 0, -1]]) / np.math.sqrt(2)
            F1 = cv.sepFilter2D(src=img, ddepth=cv.CV_64F, kernelX=M1, kernelY=G)
            F2 = scipy.ndimage.convolve(img, M2, mode='nearest')
            F3 = scipy.ndimage.convolve(img, M3, mode='nearest')
            F4 = cv.sepFilter2D(src=img, ddepth=cv.CV_64F, kernelX=M1, kernelY=G)
            FM = np.abs(F1) + np.abs(F2) + np.abs(F3) + np.abs(F4)
            return cv.mean(FM)[0]

        def curvature(self, img):
            """Curvature (CURV) Helmli 2001
            """
            M1 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            M2 = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
            P0 = scipy.ndimage.convolve(img, M1, mode='nearest') / 6
            P1 = scipy.ndimage.convolve(img, M1, mode='nearest') / 6
            P2 = 3 * scipy.ndimage.convolve(img, M2, mode='nearest') / 10. - scipy.ndimage.convolve(img, M2,
                                                                                                    mode='nearest') / 5.
            P3 = -scipy.ndimage.convolve(img, M2, mode='nearest') / 5. + 3 * scipy.ndimage.convolve(img, M2,
                                                                                                    mode='nearest') / 10.
            FM = np.abs(P0) + np.abs(P1) + np.abs(P2) + np.abs(P3)
            return cv.mean(FM)[0]

        '''
        def gaussianDerivative(self,img):
            """GDER Gaussian derivative (Geusebroek2000)"""
            N = np.math.floor(img.shape[1]/2.)
            sig = N/2.5
            x, y = np.mgrid[-N:N, -N:N]
            print(x,y)
            G = np.math.exp(-(x*x+y*y)/(2*sig*2*sig))/(2*np.pi*sig)
            Gx = -x*G/(sig*sig)
            Gy = -y*G/(sig*sig)
            Rx = scipy.ndimage.convolve(img, Gx, mode='nearest')
            Ry = scipy.ndimage.convolve(img, Gy, mode='nearest')
            FM = Gx * Gx + Gy * Gy
            mn = cv.mean(FM)[0]
            if np.isnan(mn):
                return np.nanmean(FM)
            return mn
        '''

        def brenner(self, img):
            """Brenner Gradient (BGR) Brenner97
            """
            shape = np.shape(img)
            out = 0
            for y in range(0, shape[1]):
                for x in range(0, shape[0] - 2):
                    out += (int(img[x + 2, y]) - int(img[x, y])) ** 2

            return out

        def sumModulusDifference(self, img):
            """Sum Modulus Difference (SMD)
            """
            shape = np.shape(img)
            out = 0
            for y in range(0, shape[1] - 1):
                for x in range(0, shape[0] - 1):
                    out += math.fabs(int(img[x, y]) - int(img[x, y - 1]))
                    out += math.fabs(int(img[x, y] - int(img[x + 1, y])))
            return out

        def sumModulusDifference2(self, img):
            """Sum Modulus Difference (SMD) 2
            """
            shape = np.shape(img)
            out = 0
            for y in range(0, shape[1] - 1):
                for x in range(0, shape[0] - 1):
                    out += math.fabs(int(img[x, y]) - int(img[x + 1, y])) * math.fabs(
                        int(img[x, y] - int(img[x, y + 1])))
            return out

        def energygradient(self, img):
            """Energy Gradient (EGR)
            """
            shape = np.shape(img)
            out = 0
            for y in range(0, shape[1] - 1):
                for x in range(0, shape[0] - 1):
                    out += ((int(img[x + 1, y]) - int(img[x, y])) ** 2) * ((int(img[x, y + 1] - int(img[x, y]))) ** 2)
            return out

        def vollathsautocorrelation(self, img):
            """Vollathsautocorrelation (VCR)
            """
            shape = np.shape(img)
            u = np.mean(img)
            out = -shape[0] * shape[1] * (u ** 2)
            for y in range(0, shape[1]):
                for x in range(0, shape[0] - 1):
                    out += int(img[x, y]) * int(img[x + 1, y])
            return out

        def entropy(self, img):
            """Entropy (EHS)
            """
            [rows, cols] = img.shape
            h = 0
            hist_gray = cv.calcHist([img], [0], None, [256], [0.0, 255.0])
            # hn valueis not correct
            hb = np.zeros((256, 1), np.float32)
            # hn = np.zeros((256, 1), np.float32)
            for j in range(0, 256):
                hb[j, 0] = hist_gray[j, 0] / (rows * cols)
            for i in range(0, 256):
                if hb[i, 0] > 0:
                    h = h - (hb[i, 0]) * math.log(hb[i, 0], 2)
            out = h
            return out

    class ImageAlignment:
        @staticmethod
        def NCC(img1, img2, verbose=True):
            """Image alignment using normalized cross correlation (NCC)
            Images should be of the same size

            :Parameters: image1, image2
            :Returns: image, shiftx, shifty
            """

            # img1b=img1
            # img2b=img2

            # img1b = (img1 / 256).astype('uint8')
            # img2b = (img2 / 256).astype('uint8')
            img1b = ims.Image.Convert.toGray(img1)
            img2b = ims.Image.Convert.toGray(img2)
            img1b = ims.Image.Convert.to8bit(img1b)
            img2b = ims.Image.Convert.to8bit(img2b)

            t_center_perc = [0.5, 0.5]
            t_size_perc = [0.7, 0.7]

            find_feature = ims.Analyze.FindFeature()  # create object
            find_feature.create_template(img1b, t_center_perc, t_size_perc)
            # find_feature.set_searchregion_as_template_perc(1.2) #hard to adjust for full image, use full image instead
            find_feature.run(img2b, verbose=True)

            shift_in_pixels = find_feature.shift_in_pixels
            score = find_feature.score
            shiftx = shift_in_pixels[0] * -1.0
            shifty = shift_in_pixels[1] * -1.0

            img3 = ims.Image.Transform.translate(img2, shiftx, shifty)
            if verbose == True:
                print("NCC shift x {}, shift y {}, score {:.4f}".format(shiftx, shifty, score))
            return img3, shiftx, shifty, score

        @staticmethod
        def CenterOfMass(img1, img2, verbose=True):
            """Image alignment using center of mass
            Images should be of the same size

            :Parameters: image1, image2
            :Returns: image, shiftx, shifty
            """

            img1b = ims.Image.Convert.to8bit(img1)
            img2b = ims.Image.Convert.to8bit(img2)

            img, x0, y0 = ims.Analyze.find_image_center_of_mass(img1b, verbose=False)
            img, x1, y1 = ims.Analyze.find_image_center_of_mass(img2b, verbose=False)
            # print(x0, y0, x1, y1)
            shiftx = x1 - x0
            shifty = (y1 - y0)
            score = 1  # score not applicable

            img3 = ims.Image.Transform.translate(img2, shiftx, shifty)
            if verbose == True:
                print("COM shift x {}, shift y {}, score {:.4f}".format(shiftx, shifty, score))
            return img3, shiftx, shifty, score

        @staticmethod
        def GRAD(img1, img2, verbose=True):
            """Image alignment by performing a gradient prior to applying a normalized cross correlation
            Images should be of the same size

            :Parameters: image1, image2
            :Returns: image, shiftx in pixels, shifty in pixels
            """

            # hog method
            img1b = ims.Image.Convert.to8bit(img1)
            # img1b = ims.Image.histostretch_equalized(img1b)
            img1b, angle = ims.Image.Process.gradient_image(img1b)
            # img1b,angle = ims.Image.gradient_image_nonmaxsuppressed(img1b,31,0)
            # img1b = ims.Image.Convert.to8bit(img1b)

            img2b = ims.Image.Convert.to8bit(img2)
            # img2b = ims.Image.histostretch_equalized(img2b)
            img2b, angle = ims.Image.Process.gradient_image(img2b)
            # img2b,angle = ims.Image.gradient_image_nonmaxsuppressed(img2b,31,0)
            # img2b = ims.Image.Convert.to8bit(img2b)

            # ims.View.plot4(img1b,img3b,img2b,img4b,"gradientemp","nonmaxtemp","gradienttarget","nonmaxtarget")

            t_center_perc = [0.5, 0.5]
            t_size_perc = [0.8, 0.8]

            find_feature = ims.Analyze.FindFeature()  # create object
            find_feature.create_template(img1b, t_center_perc, t_size_perc)
            # find_feature.set_searchregion_as_template_perc(1)
            find_feature.run(img2b, verbose=True)

            shift_in_pixels = find_feature.shift_in_pixels
            score = find_feature.score
            shiftx = shift_in_pixels[0] * -1.0
            shifty = shift_in_pixels[1] * -1.0
            img3 = ims.Image.Transform.translate(img2, shiftx, shifty)
            if verbose == True:
                print("GRAD shift x {}, shift y {}, score {:.4f}".format(shiftx, shifty, score))
            return img3, shiftx, shifty, score

        @staticmethod
        def NONMAX(img1, img2, verbose=True):
            """Image alignment using gradients in combination with non-maximum suppression prior to applying a normalized cross correlation
            Images should be of the same size

            :Parameters: image1, image2
            :Returns: image, shiftx, shifty
            """

            # hog method
            img1b = ims.Image.Convert.to8bit(img1)
            # img1b = ims.Image.histostretch_equalized(img1b)
            # img1b, angle = ims.Image.gradient_image(img1b)
            img1b, angle = ims.Image.Process.gradient_image_nonmaxsuppressed(img1b, 31, 0)
            img1b = ims.Image.Convert.to8bit(img1b)

            img2b = ims.Image.Convert.to8bit(img2)
            # img2b = ims.Image.histostretch_equalized(img2b)
            # img2b, angle = ims.Image.gradient_image(img2b)
            img2b, angle = ims.Image.Process.gradient_image_nonmaxsuppressed(img2b, 31, 0)
            img2b = ims.Image.Convert.to8bit(img2b)

            # ims.View.plot4(img1b,img3b,img2b,img4b,"gradientemp","nonmaxtemp","gradienttarget","nonmaxtarget")

            t_center_perc = [0.5, 0.5]
            t_size_perc = [0.8, 0.8]

            find_feature = ims.Analyze.FindFeature()  # create object
            find_feature.create_template(img1b, t_center_perc, t_size_perc)
            # find_feature.set_searchregion_as_template_perc(1)
            find_feature.run(img2b, verbose=True)

            shift_in_pixels = find_feature.shift_in_pixels
            score = find_feature.score
            shiftx = shift_in_pixels[0] * -1.0
            shifty = shift_in_pixels[1] * -1.0
            img3 = ims.Image.Transform.translate(img2, shiftx, shifty)
            if verbose == True:
                print("NONMAX shift x {}, shift y {}, score {:.4f}".format(shiftx, shifty, score))
            return img3, shiftx, shifty, score

        @staticmethod
        def ORB(img1, img2, verbose=True):
            """Image alignment of a source and target image using ORB (Oriented Fast and Rotated BRIEF)
            Images should be of the same size

            :Parameters: image1, image2
            :Returns: image, shiftx, shifty
            """

            # Initiate ORB detector
            orb = cv.ORB_create(nfeatures=1000, scoreType=cv.ORB_HARRIS_SCORE)

            # Find the keypoints and descriptors with ORB
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)

            # BFMatcher with default params
            bf = cv.BFMatcher()

            try:
                # Apply ratio test
                pairMatches = bf.knnMatch(des1, des2, k=2)
                rawMatches = []
                for m, n in pairMatches:
                    if m.distance < 0.7 * n.distance:
                        rawMatches.append(m)

                sortMatches = sorted(rawMatches, key=lambda x: x.distance)
                matches = sortMatches[0:128]

                image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
                image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)

                lnxlist = []
                lnylist = []
                for i in range(0, len(matches)):
                    image_1_points[i] = kp1[matches[i].queryIdx].pt
                    image_2_points[i] = kp2[matches[i].trainIdx].pt

                    x0 = image_1_points[i][0][0]
                    y0 = image_1_points[i][0][1]
                    x1 = image_2_points[i][0][0]
                    y1 = image_2_points[i][0][1]

                    lnx = x0 - x1
                    lny = y0 - y1
                    lnxlist.append(lnx)
                    lnylist.append(lny)

                lnxfin = np.median(lnxlist)
                lnyfin = np.median(lnylist)
                score = 1
            except:
                print("failed to match.")
                score = 0
                lnxfin = 0
                lnyfin = 0
            im_out = ims.Image.Transform.translate(img2, lnxfin, lnyfin)
            if verbose == True:
                print("ORB shift x {}, shift y {}, score {:.4f}".format(lnxfin, lnyfin, score))
            return im_out, lnxfin, lnyfin, score

    class FindFeature(object):
        def __init__(self):
            self._shift_in_pixels = [0, 0]
            self._score = 0
            self._t_center_perc = [0, 0]
            self._t_size_perc = [0, 0]
            self._s_center_perc = [0.5, 0.5]
            self._s_size_perc = [1, 1]
            self._template = np.zeros([128, 128, 3], dtype=np.uint8)
            self._t_center_pixels = [0, 0]
            self._t_size_pixels = [0, 0]
            self._s_center_pixels = [0, 0]
            self._s_size_pixels = [0, 0]
            self._template_matcher = 3  # 3=ccorr_normed, 5=ccoef_normed
            self._apply_gradient = False

        @property
        def shift_in_pixels(self):
            return self._shift_in_pixels

        @property
        def score(self):
            return self._score

        @property
        def t_center_perc(self):
            return self._t_center_perc

        @property
        def t_size_perc(self):
            return self._t_size_perc

        @property
        def s_center_perc(self):
            return self._s_center_perc

        @property
        def s_size_perc(self):
            return self._s_size_perc

        @property
        def template(self):
            return self._template

        @property
        def t_center_pixels(self):
            return self._t_center_pixels

        @property
        def t_size_pixels(self):
            return self._t_size_pixels

        @property
        def s_center_pixels(self):
            return self._s_center_pixels

        @property
        def s_size_pixels(self):
            return self._s_size_pixels

        @property
        def template_matcher(self):
            return self._template_matcher

        @property
        def apply_gradient(self):
            return self._apply_gradient

        @t_center_perc.setter
        def t_center_perc(self, value):
            self._t_center_perc = value

        @t_size_perc.setter
        def t_size_perc(self, value):
            self._t_size_perc = value

        @s_center_perc.setter
        def s_center_perc(self, value):
            self._s_center_perc = value

        @s_size_perc.setter
        def s_size_perc(self, value):
            self._s_size_perc = value

        @template.setter
        def template(self, value):
            self._template = value

        @t_center_pixels.setter
        def t_center_pixels(self, value):
            self._t_center_pixels = value

        @t_size_pixels.setter
        def t_size_pixels(self, value):
            self._t_size_pixels = value

        @s_center_pixels.setter
        def s_center_pixels(self, value):
            self._s_center_pixels = value

        @s_size_pixels.setter
        def s_size_pixels(self, value):
            self._s_size_pixels = value


        #methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
        #                   'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED','zncc','ssd','sad']
        @template_matcher.setter
        def template_matcher(self, value):
            self._template_matcher = value

        @apply_gradient.setter
        def apply_gradient(self, value):
            self._apply_gradient = value

        def __cut(self, img, center=[0, 0], size=[0, 0]):
            """Cut out image to create a template for template matching

            :parameters: image, center=[0, 0], size=[0, 0]
            :Returns: image
            """
            template = ims.Image.cut(img, center, size)
            return template

        def __crop(self, img, x0, y0, x1, y1):
            res = ims.Image.crop(img, x0, y0, x1, y1)
            return res

        def run(self, img, verbose=True):
            """Start find feature routine
            based on OpenCV template matcher, returns shift in pixels and score

            :parameters: image, verbose
            :Returns: -
            """

            template = ims.Image.Convert.toGray(self.template)
            template = ims.Image.Convert.to8bit(template)
            if self.apply_gradient == True:
                template, angle = ims.Image.Process.gradient_image(template)

            img = ims.Image.Convert.toGray(img)
            img = ims.Image.Convert.to8bit(img)
            if self.apply_gradient == True:
                img, angle = ims.Image.Process.gradient_image(img)

            # crop to region and adjust template
            self.s_center_pixels, self.s_size_pixels = ims.Analyze.rectangle_percentage_to_pixels(img,
                                                                                                  self.s_center_perc,
                                                                                                  self.s_size_perc)
            img1 = self.__cut(img, self.s_center_pixels, self.s_size_pixels)

            offnew = [-(self.s_center_pixels[0] - int(self.s_size_pixels[0] / 2)),
                      -(self.s_center_pixels[1] - int(self.s_size_pixels[1] / 2))]

            self.t_center_pixels, self.t_size_pixels = ims.Analyze.rectangle_percentage_to_pixels(img,
                                                                                                  self.t_center_perc,
                                                                                                  self.t_size_perc)  # determine center of cutout template
            templateorigin = [(self.t_center_pixels[0] + offnew[0]), (self.t_center_pixels[1] + offnew[1])]

            method = self.template_matcher
            showresult = False

            pt, score = self.__templatematch(img1, template, templateorigin, method)
            self._shift_in_pixels = pt
            self._score = score
            if verbose == True:
                print(
                    "locate_feature: shift in pixels {}, score {:.4f}".format(pt, score))

        # @staticmethod
        def create_template(self, img, t_center_perc, t_size_perc):
            """Create template for find feature
            template is saved as property in FindFeature class

            :Parameters: image, t_center_perc, t_size_perc
            :Returns: -
            """
            t_center_px, t_size_px = ims.Analyze.rectangle_percentage_to_pixels(img, t_center_perc, t_size_perc)

            template = self.__cut(img, t_center_px, t_size_px)
            self.template = template
            self.t_center_perc = t_center_perc
            self.t_size_perc = t_size_perc
            self.t_center_pixels = t_center_px
            self.t_size_pixels = t_size_px
            self.original_feature_center = t_center_perc

        # @staticmethod
        def create_template_dialog(self, img0):
            """Create template for find feature using an interactive dialog
            template is saved as property in FindFeature class

            :Parameters: image
            :Returns: -
            """
            img0 = ims.Image.Convert.to8bit(img0)
            img2 = ims.Analyze.add_text(img0, 0, 0, 'Input: drag a rectangle around the fiducial area.', 20)
            shapes = ims.Dialogs.select_areas(img2, 'Template matching input')
            if not shapes:
                print("Warning: Shape is not defined, taking 90% of full image")
                shapes = [[(int(img0.shape[1]*0.1),int(img0.shape[0]*0.1)),(int(img0.shape[1]*0.9),int(img0.shape[0]*0.9))]]
            if len(shapes)>1:
                print("Warning: Multiple shapes defined, taking 90% of full image")
                shapes = [[(int(img0.shape[1]*0.1),int(img0.shape[0]*0.1)),(int(img0.shape[1]*0.9),int(img0.shape[0]*0.9))]]
            widthperc, heightperc, centerxperc, centeryperc = ims.Analyze.rectangle_pixels_to_percentage(img0,
                                                                                                         shapes[0])
            t_center_perc = [centerxperc, centeryperc]
            t_size_perc = [widthperc, heightperc]
            # template = self.create_template(img0, t_center_perc, t_size_perc)
            t_center_px, t_size_px = ims.Analyze.rectangle_percentage_to_pixels(img0, t_center_perc, t_size_perc)
            template = self.__cut(img0, t_center_px, t_size_px)
            self.template = template
            self.t_center_perc = t_center_perc
            self.t_size_perc = t_size_perc
            self.t_center_pixels = t_center_px
            self.t_size_pixels = t_size_px
            self.original_feature_center = self.t_center_perc

        # standard routine provides output in pixels
        def __templatematch(self, img, template, templateposition, method):
            pt, score = self.__templatematchcore(img, template, method)
            x1 = templateposition[0]
            y1 = templateposition[1]
            xshiftp = pt[0] - x1
            yshiftp = pt[1] - y1
            ptp = (xshiftp, yshiftp)
            # print("Shift in px:" , ptp)
            return ptp, score

        def __centeroftemplate(self, img, x0perc, y0perc, x1perc, y1perc):
            # topleft 0,0; bottomright 100,100
            w0, h0 = img.shape[1::-1]
            x0p = int(w0 * x0perc / 100)
            y0p = int(h0 * y0perc / 100)
            x1p = int(w0 * x1perc / 100)
            y1p = int(h0 * y1perc / 100)
            xp = int(x0p + (x1p - x0p) / 2)
            yp = int(y0p + (y1p - y0p) / 2)
            print("CenterOfTemplate", xp, yp)
            return (xp, yp)

        def __templatematchcore(self, img, template, method):
            w, h = template.data.shape[1::-1]

            # All the 6 methods for comparison in a list
            # Apply template Matching

            # ims.View.plot_list([img,template])
            if (method < 6):

                methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                           'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

                meth = methods[method]
                methodn = eval(meth)

                res = cv.matchTemplate(img, template, methodn)
                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                    top_left = min_loc
                else:
                    top_left = max_loc
            else:
                if (method == 6):
                    top_left, max_val = self._template_matching_zncc(img, template)
                if (method == 7):
                    top_left, max_val = self._template_matching_ssd(img, template)
                if (method == 8):
                    top_left, max_val = self._template_matching_sad(img, template)

            #print(top_left, max_val)
            # sys.exit()
            bottom_right = (w, h)
            wd2 = int(w / 2)
            hd2 = int(h / 2)
            pt = (top_left[0] + wd2, top_left[1] + hd2)
            # print("Position found:",pt)
            score = max_val
            return pt, score

        def _template_matching_ssd(self, src, temp):
            # https://github.com/PrinzEugen7/Lesson/tree/master/Python/opencv/image/template-matching
            # for evaluation
            h, w = src.shape
            ht, wt = temp.shape

            score = np.empty((h - ht, w - wt))

            for dy in range(0, h - ht):
                for dx in range(0, w - wt):
                    diff = (src[dy:dy + ht, dx:dx + wt] - temp) ** 2
                    score[dy, dx] = diff.sum()

            pt = np.unravel_index(score.argmin(), score.shape)
            scoreout = score.argmax()
            return (pt[1], pt[0]), scoreout

        def _template_matching_zncc(self, src, temp):
            # https://github.com/PrinzEugen7/Lesson/tree/master/Python/opencv/image/template-matching
            # for evaluation
            h, w = src.shape
            ht, wt = temp.shape

            score = np.empty((h - ht, w - wt))

            src = np.array(src, dtype="float")
            temp = np.array(temp, dtype="float")

            mu_t = np.mean(temp)

            for dy in range(0, h - ht):
                for dx in range(0, w - wt):
                    roi = src[dy:dy + ht, dx:dx + wt]
                    mu_r = np.mean(roi)
                    roi = roi - mu_r
                    temp = temp - mu_t

                    num = np.sum(roi * temp)
                    den = np.sqrt(np.sum(roi ** 2)) * np.sqrt(np.sum(temp ** 2))
                    if den == 0: score[dy, dx] = 0
                    score[dy, dx] = num / den

            pt = np.unravel_index(score.argmin(), score.shape)
            scoreout = score.argmax()
            return (pt[1], pt[0]), scoreout

        def _template_matching_sad(self, src, temp):
            # https://github.com/PrinzEugen7/Lesson/tree/master/Python/opencv/image/template-matching
            # for evaluation
            h, w = src.shape
            ht, wt = temp.shape

            score = np.empty((h - ht, w - wt))

            for dy in range(0, h - ht):
                for dx in range(0, w - wt):
                    diff = np.abs(src[dy:dy + ht, dx:dx + wt] - temp)
                    score[dy, dx] = diff.sum()

            pt = np.unravel_index(score.argmin(), score.shape)
            scoreout = score.argmax()
            return (pt[1], pt[0]), scoreout

        def plot_matchresult(self, img, verbose=False):
            """plot matchresult of find feature showing template and search region

            :Parameters: img, verbose
            :Returns: image
            """

            # pt = featurelocation_center_in_pixels
            w0, h0 = img.shape[1::-1]
            w1, h1 = self.template.shape[1::-1]
            t_size = [(w1 / w0), (h1 / h0)]

            xoff = self.shift_in_pixels[0]
            yoff = self.shift_in_pixels[1]

            xoffperc = self.t_center_perc[0] + (xoff / w0)
            yoffperc = self.t_center_perc[1] + (yoff / h0)

            self.t_center_perc = [xoffperc, yoffperc]
            rgb1 = self.plot_searchregion_and_template(img, verbose=verbose)
            return rgb1

        def plot_searchregion_and_template(self, img, verbose=False, autoclose=0):
            """plot search region and template for find feature

            :Parameters: img, verbose
            :Returns: image
            """
            w, h = img.shape[1::-1]

            xt0 = ims.Misc.multipleof2((self.t_center_perc[0] - self.t_size_perc[0] * 0.5) * w)
            yt0 = ims.Misc.multipleof2((self.t_center_perc[1] - self.t_size_perc[1] * 0.5) * h)
            xt1 = ims.Misc.multipleof2((self.t_center_perc[0] + self.t_size_perc[0] * 0.5) * w)
            yt1 = ims.Misc.multipleof2((self.t_center_perc[1] + self.t_size_perc[1] * 0.5) * h)

            xs0 = ims.Misc.multipleof2((self.s_center_perc[0] - self.s_size_perc[0] * 0.5) * w)
            ys0 = ims.Misc.multipleof2((self.s_center_perc[1] - self.s_size_perc[1] * 0.5) * h)
            xs1 = ims.Misc.multipleof2((self.s_center_perc[0] + self.s_size_perc[0] * 0.5) * w)
            ys1 = ims.Misc.multipleof2((self.s_center_perc[1] + self.s_size_perc[1] * 0.5) * h)

            # rgb = img.copy()

            img = ims.Image.Convert.to8bit(img)
            try:
                rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
            except:
                rgb = img.copy()

            rgb = cv.rectangle(rgb, (xt0, yt0), (xt1, yt1), (0, 255, 0), 2)
            rgb = cv.rectangle(rgb, (xs0, ys0), (xs1, ys1), (255, 255, 0), 2)
            if verbose == True:
                print("template: {},{},{},{} searchregion: {},{},{},{}".format(xt0, yt0, xt1, yt1, xs0, ys0, xs1, ys1))
                rgb1 = ims.Image.Convert.BGRtoRGB(rgb)
                plt.plot()
                plt.gcf().canvas.set_window_title('Find Feature')
                plt.imshow(rgb1)

                if autoclose > 0:
                    try:
                        plt.show(block=False)
                        plt.pause(autoclose)  # 3 seconds, I use 1 usually
                    except:
                        print("interrupted while waiting.")
                    plt.close("all")
                else:
                    plt.show()

            return rgb

        def set_searchregion_as_template_perc(self, img, perc=1.2, verbose=False):
            """set searchregion as percentage of template size for find feature

            :Parameters: img, percentage
            :Returns: -
            """
            self.s_center_perc = self.t_center_perc
            self.s_size_perc = [self.t_size_perc[0] * perc, self.t_size_perc[1] * perc]
            if self.s_size_perc[0] > 1:
                self.s_size_perc[0] = 1
            if self.s_size_perc[1] > 1:
                self.s_size_perc[1] = 1
            if verbose == True:
                print('searchregion, searchsize: ', self.s_center_perc, self.s_size_perc)
            #calculate same but in pixels
            self.s_center_pixels, self.s_size_pixels = ims.Analyze.rectangle_percentage_to_pixels(img,
                                                                                                  self.s_center_perc,
                                                                                                  self.s_size_perc)
