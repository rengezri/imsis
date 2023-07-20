#!/usr/bin/env python

"""
This file is part of IMSIS
Licensed under the MIT license:
http://www.opensource.org/licenses/MIT-license

This module contains image processing methods
"""

import os
import sys

import cv2 as cv
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.misc
from matplotlib import pyplot as plt
import numpy.random as random
from matplotlib.colors import hsv_to_rgb
from datetime import datetime
from scipy import ndimage

import imsis


class Image(object):

    @staticmethod
    def load(filename, verbose=True):
        """Load image
        Supported file formats: PNG, TIF, BMP
        note: by default images are converted to grayscale (8bit gray), conversion to 8 bit can be disabled.

        :Parameters: filename, gray=True, verbose=False
        :Returns: image
        """

        img = None
        if (os.path.isfile(filename)):
            img = cv.imread(filename, -1)
            if (verbose == True):
                print("Load file:", filename, img.shape, img.dtype)
        else:
            print('Error, image file does not exist. ', filename)
            sys.exit()
        try:
            q = img.shape
        except:
            print('Error, image file could not be read. ', filename)
            sys.exit()
        return img

    @staticmethod
    def crop_rectangle(img, rect):
        """Crop an image using rectangle shape as input [(x0,y0),(x1,y1)]

        :Parameters: image, rectangle
        :Returns: image
        """
        if len(rect) > 0:
            out = Image.crop(img, rect[0][0], rect[0][1], rect[1][0], rect[1][1])
        else:
            print("Error: rectangle not defined.")
            out = img
        return out

    @staticmethod
    def crop(img, x0, y0, x1, y1):
        """Crop an image using pixels at x0,y0,x1,y1

        :Parameters: image, x0, y0, x1, y1
        :Returns: image
        """
        res = img[y0:y1, x0:x1]  # Crop from y0:y1,x0:x1
        # print("Cropped region: (" , x0,y0,x1,y1,")")
        return res

    @staticmethod
    def crop_percentage(img, scale=1.0):
        """Crop an image centered

        :Parameters: image, scale=1.0
        :Returns: image
        """
        center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
        width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
        left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
        top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
        img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
        return img_cropped

    @staticmethod
    def resize(img, factor=0.5):
        """Resize image

        :Parameters: image, factor
        :Returns: image
        """
        small = cv.resize(img, (0, 0), fx=factor, fy=factor)
        return small

    @staticmethod
    def _blur_edge(img, d=31):
        """blur edge

        :Parameters: image, d
        :Returns: image
        """

        h, w = img.shape[:2]
        img_pad = cv.copyMakeBorder(img, d, d, d, d, cv.BORDER_WRAP)
        img_blur = cv.GaussianBlur(img_pad, (2 * d + 1, 2 * d + 1), -1)[d:-d, d:-d]
        y, x = np.indices((h, w))
        dist = np.dstack([x, w - x - 1, y, h - y - 1]).min(-1)
        w = np.minimum(np.float32(dist) / d, 1.0)
        return img * w + img_blur * (1 - w)

    @staticmethod
    def _motion_kernel(angle, d, sz=65):
        """determine motion kernel value

        :Parameters: angle, d, size
        :Returns: kernel
        """
        kern = np.ones((1, d), np.float32)
        c, s = np.cos(angle), np.sin(angle)
        A = np.float32([[c, -s, 0], [s, c, 0]])
        sz2 = sz // 2
        A[:, 2] = (sz2, sz2) - np.dot(A[:, :2], ((d - 1) * 0.5, 0))
        kern = cv.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
        return kern

    @staticmethod
    def _defocus_kernel(d, sz=65):
        """determine defocus kernel value

        :Parameters: d, size
        :Returns: kernel
        """
        kern = np.zeros((sz, sz), np.uint8)
        cv.circle(kern, (sz, sz), d, 255, -1, cv.LINE_AA, shift=1)
        kern = np.float32(kern) / 255.0
        return kern

    @staticmethod
    def _image_stats(image):
        # compute the mean and standard deviation of each channel
        (l, a, b) = cv.split(image)
        (lMean, lStd) = (l.mean(), l.std())
        (aMean, aStd) = (a.mean(), a.std())
        (bMean, bStd) = (b.mean(), b.std())

        # return the color statistics
        return (lMean, lStd, aMean, aStd, bMean, bStd)

    @staticmethod
    def save(img, fn, verbose=True):
        """Save image (PNG,TIF)

        :Parameters: image, filename
        """
        try:
            if (os.path.dirname(fn)):
                os.makedirs(os.path.dirname(fn), exist_ok=True)  # mkdir if not empty
            cv.imwrite(fn, img)
            if verbose == True:
                print("file saved. ", fn)
        except:
            print("Error: cannot save file {}".format(fn))

    @staticmethod
    def save_withuniquetimestamp(img):
        """Save PNG image with unique timestamp.

        :Parameters: image
        """
        path = "./output/"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sttime = datetime.now().strftime('Image_%Y%m%d%H%M%S')
        fn = path + sttime + '.png'
        print("file saved. ", fn)
        cv.imwrite(fn, img)

    # implemented twice remove the 2nd one
    @staticmethod
    def cut(img, center=[0, 0], size=[0, 0]):
        """return a image cut out

        :Parameters: image, center=[0, 0], size=[0, 0]
        :Returns: image
        """
        x0 = center[0] - round(size[0] * 0.5)
        x1 = center[0] + round(size[0] * 0.5)
        y0 = center[1] - round(size[1] * 0.5)
        y1 = center[1] + round(size[1] * 0.5)

        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        template = Image.crop(img, int(x0), int(y0), int(x1), int(y1))
        return template

    @staticmethod
    def _multipleof2(number):
        """Rounds the given number to the nearest multiple of two."""
        remainder = number % 2
        if remainder > 1:
            number += (2 - remainder)
        else:
            number -= remainder
        return int(number)

    @staticmethod
    def subtract(img0, img1):
        """subtract 2 images

        :Parameters: image1, image2
        :Returns: image
        """
        out = cv.subtract(img0, img1)
        return out

    @staticmethod
    def add(img0, img1, alpha=0.5):
        """add 2 images weighted (default alpha=0.5)
        :Parameters: image1, image2, alpha
        :Returns: image
        """
        a = img0
        b = img1
        beta = 1 - alpha
        gamma = 0
        out = cv.addWeighted(a, alpha, b, beta, gamma)
        return out

    def add2(img0, img1, alpha=0.5):
        """add 2 images weighted (default alpha=0.5) even if size is slightly different
        :Parameters: image1, image2, alpha
        :Returns: image
        """
        bg_img = img0
        fg_img = img1

        w, h, _ = bg_img.shape
        w1, h1, _ = fg_img.shape
        if w1 > w:
            w = w1
        if h1 > h:
            h = h1
        bg_img1 = np.zeros((w, h, 3), np.uint8)
        fg_img1 = np.zeros((w, h, 3), np.uint8)
        bg_img1[0: 0 + bg_img.shape[0], 0: 0 + bg_img.shape[1]] = bg_img
        fg_img1[0: 0 + fg_img.shape[0], 0: 0 + fg_img.shape[1]] = fg_img
        a = bg_img1
        b = fg_img1
        beta = 1 - alpha
        gamma = 0
        out = cv.addWeighted(a, alpha, b, beta, gamma)
        return out

    @staticmethod
    def add_overlay(bg_img, fg_img):
        """add 2 images, foregrond image with black as mask overlays a background image with black as mask
        :Parameters: image1, image2
        :Returns: image
        """

        w, h, _ = bg_img.shape
        w1, h1, _ = fg_img.shape
        if w1 > w:
            w = w1
        if h1 > h:
            h = h1
        bg_img1 = np.zeros((w, h, 3), np.uint8)
        fg_img1 = np.zeros((w, h, 3), np.uint8)
        bg_img1[0: 0 + bg_img.shape[0], 0: 0 + bg_img.shape[1]] = bg_img
        fg_img1[0: 0 + fg_img.shape[0], 0: 0 + fg_img.shape[1]] = fg_img

        # create a mask of the foreground image
        fg_mask = cv.cvtColor(fg_img1, cv.COLOR_BGR2GRAY)
        _, fg_mask = cv.threshold(fg_mask, 1, 255, cv.THRESH_BINARY)

        # invert the mask so that the foreground is white and the background is black
        fg_mask_inv = cv.bitwise_not(fg_mask)

        # apply the mask to the foreground and background images
        fg = cv.bitwise_and(fg_img1, fg_img1, mask=fg_mask)
        bg = cv.bitwise_and(bg_img1, bg_img1, mask=fg_mask_inv)

        # combine the foreground and background images
        result = cv.add(fg, bg)
        return result

    @staticmethod
    def new(height, width):
        """Create a new blank image

        :Parameters: height,width
        :Returns: image
        """
        img = np.zeros((height, width), np.uint8)
        return img

    @staticmethod
    def gaussiankernel(kernlen=21, nsig=3):
        """returns a 2D gaussian kernel

        :Parameters: kernelsize, nsig
        :Returns: image
        """
        x = np.linspace(-nsig, nsig, kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        return kern2d / kern2d.sum()

    @staticmethod
    def info(img):
        """get image properties

        :Parameters: img
        """
        print("{}, {}, {}".format(img.shape, img.size, img.dtype))

    @staticmethod
    def unique_colours(image):
        """get number of unique colors in an image

        :Parameters: img
        """
        print(image.shape)
        if (len(image.shape) == 3):
            out = len(np.unique(image.reshape(-1, image.shape[2]), axis=0))
            # b, g, r = cv.split(image)
            # out_in_32U_2D = np.int32(b) << 16 + np.int32(g) << 8 + np.int32(r)  # bit wise shift 8 for each channel.
            # out_in_32U_1D = out_in_32U_2D.reshape(-1)  # convert to 1D
            # np.unique(out_in_32U_1D)
            # out = len(np.unique(out_in_32U_1D))
        else:
            out_in_32U_2D = np.int32(image)  # bit wise shift 8 for each channel.
            out_in_32U_1D = out_in_32U_2D.reshape(-1)  # convert to 1D
            np.unique(out_in_32U_1D)
            out = len(np.unique(out_in_32U_1D))
        print(out)
        return out

    @staticmethod
    def zoom(image0, factor=2, cx=0.5, cy=0.5):
        """
        zoom image, resize with factor n, crop in center to same size as original image
        :Parameters: image0, zoom factor
        :Returns: image
        """
        h = image0.shape[0]
        w = image0.shape[1]
        img = Image.resize(image0, factor)
        x0 = int(factor * w * cx * 0.5)
        y0 = int(factor * h * cy * 0.5)
        x1 = x0 + w
        y1 = y0 + h
        # print(x0, y0, x1, y1, w, h, img.shape[0], img.shape[1])
        img = Image.crop(img, x0, y0, x1, y1)
        return img

    class Process:

        @staticmethod
        def directionalsharpness(img, ksize=-1):
            """
            DirectionalSharpness

            Measure sharnpess in X and Y seperately

            Note: Negative slopes are missed when converting to unaryint8, therefore convert to float
            :Parameters: image, kernel
            :Returns: gradientx , gradienty, gradientxy, theta
            """
            sobelx64f = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=ksize)
            sobely64f = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=ksize)

            grad = np.power(np.power(sobelx64f, 2.0) + np.power(sobely64f, 2.0), 0.5)
            theta = np.arctan2(sobely64f, sobelx64f)

            Gx = np.absolute(sobelx64f)
            Gy = np.absolute(sobely64f)

            mx = cv.mean(Gx)[0]
            my = cv.mean(Gy)[0]

            return mx, my, grad, theta

        @staticmethod
        def gradient_image(img, kx=11, ky=3):
            """Create a gradient image
            Method used: gradient by bi-directional sobel filter

            :Parameters: image, blurkernelx, blurkernely
            :Returns: image
            """
            # Calculate gradient
            gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=1)
            gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=1)
            # mag, angle = cv.cartToPolar(gx, gy, angleInDegrees=True)
            blurredgx = cv.GaussianBlur(gx, (kx, ky), 1)
            blurredgy = cv.GaussianBlur(gy, (kx, ky), 1)
            mag, angle = cv.cartToPolar(blurredgx, blurredgy)
            return mag, angle

        @staticmethod
        def gradient_image_nonmaxsuppressed(img, blur=5, threshold=40):
            """Apply non maximum suppressed gradient filter sequence
            threshold not used??

            :Parameters: image, blur=5, threshold=40
            :Returns: image, angle
            """

            def nonmaxsuppression(im, grad):
                #  Non-maximum suppression
                gradSup = grad.copy()
                for r in range(im.shape[0]):
                    for c in range(im.shape[1]):
                        # Suppress pixels at the image edge
                        if r == 0 or r == im.shape[0] - 1 or c == 0 or c == im.shape[1] - 1:
                            gradSup[r, c] = 0
                            continue
                        tq = thetaQ[r, c] % 4

                        if tq == 0:  # 0 is E-W (horizontal)
                            if grad[r, c] <= grad[r, c - 1] or grad[r, c] <= grad[r, c + 1]:
                                gradSup[r, c] = 0
                        if tq == 1:  # 1 is NE-SW
                            if grad[r, c] <= grad[r - 1, c + 1] or grad[r, c] <= grad[r + 1, c - 1]:
                                gradSup[r, c] = 0
                        if tq == 2:  # 2 is N-S (vertical)
                            if grad[r, c] <= grad[r - 1, c] or grad[r, c] <= grad[r + 1, c]:
                                gradSup[r, c] = 0
                        if tq == 3:  # 3 is NW-SE
                            if grad[r, c] <= grad[r - 1, c - 1] or grad[r, c] <= grad[r + 1, c + 1]:
                                gradSup[r, c] = 0
                return gradSup

            img = Image.Convert.toGray(img)
            im = np.array(img, dtype=float)  # Convert to float to prevent clipping values

            # Gaussian Blur
            im2 = cv.GaussianBlur(im, (blur, blur), 0)
            # Find gradients
            im3h = cv.filter2D(im2, -1, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
            im3v = cv.filter2D(im2, -1, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))

            # Get gradient and direction
            grad = np.power(np.power(im3h, 2.0) + np.power(im3v, 2.0), 0.5)
            theta = np.arctan2(im3v, im3h)
            thetaQ = (np.round(theta * (5.0 / np.pi)) + 5) % 5  # Quantize direction

            gradSup = nonmaxsuppression(im, grad)
            return gradSup, thetaQ

        @staticmethod
        def nonlocalmeans(img, h=10, templatewindowsize=7, searchwindowsize=21):
            """Apply a non-local-means filter with filtering strength (h), template windowsize (blocksize), searchwindowsize
            :Parameters: image, h=10, templatewindowsize=7, searchwindowsize=21
            :Returns: image
            """

            # img = cv.pyrDown(img)
            dst = cv.fastNlMeansDenoising(img, None, h, templatewindowsize, searchwindowsize)
            return dst

        @staticmethod
        def deconvolution_wiener(img, d=3, noise=11):
            """Apply Wiener deconvolution
            grayscale images only

            :Parameters: image, d, noise
            :Returns: kernel
            """
            img = Image.Convert.toGray(img)
            noise = 10 ** (-0.1 * noise)
            img = np.float32(img) / 255.0
            IMG = cv.dft(img, flags=cv.DFT_COMPLEX_OUTPUT)
            psf = Image._defocus_kernel(d)
            psf /= psf.sum()
            psf_pad = np.zeros_like(img)
            kh, kw = psf.shape
            psf_pad[:kh, :kw] = psf
            PSF = cv.dft(psf_pad, flags=cv.DFT_COMPLEX_OUTPUT, nonzeroRows=kh)
            PSF2 = (PSF ** 2).sum(-1)
            iPSF = PSF / (PSF2 + noise)[..., np.newaxis]
            RES = cv.mulSpectrums(IMG, iPSF, 0)
            res = cv.idft(RES, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
            res = np.roll(res, -kh // 2, 0)
            res = np.roll(res, -kw // 2, 1)
            return res

        @staticmethod
        def median(image, kernel=5):
            """Apply a median filter

            :Parameters: image
            :Returns: image
            """
            out = cv.medianBlur(image, kernel)
            return out

        @staticmethod
        def cannyedge_auto(image, sigma=0.33):
            """Apply a Canny Edge filter automatically

            :Parameters: image, sigma
            :Returns: image
            """
            # compute the median of the single channel pixel intensities
            v = np.median(image)
            # apply automatic Canny edge detection using the computed median
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            edged = cv.Canny(image, lower, upper)
            return edged

        # smooth, threshold
        @staticmethod
        def gaussian_blur(img, smooth=3):
            """Gaussian blur image with kernel n

            :Parameters: image, kernel
            :Returns: image
            """
            # img = cv.pyrDown(img)
            imout = cv.GaussianBlur(img, (smooth, smooth), 0)
            return imout

        @staticmethod
        def unsharp_mask(img, kernel_size=5, sigma=1.0, amount=1.0, threshold=0):
            """Unsharp mask filter

            :Parameters: image, kernel_size=5, sigma=1.0, amount=1.0, threshold=0
            :Returns: image
            """
            blurred = cv.GaussianBlur(img, (5, 5), sigma)
            sharpened = float(amount + 1) * img - float(amount) * blurred
            sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
            sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
            sharpened = sharpened.round().astype(np.uint8)
            if threshold > 0:
                low_contrast_mask = np.absolute(img - blurred) < threshold
                np.copyto(sharpened, img, where=low_contrast_mask)
            return sharpened

        @staticmethod
        def FFT(img):
            """Apply a fourier transform
            generate a discrete fourier transform shift matrix and a magnitude spectrum image for viewing

            :Parameters: image
            :Returns: dft_shift, specimage
            """

            # img = Image.Convert.toGray(img)
            # do dft saving as complex output
            dft = np.fft.fft2(img, axes=(0, 1))
            # apply shift of origin to center of image
            dft_shift = np.fft.fftshift(dft)

            mag = np.abs(dft_shift)
            spec = np.log(mag) / 20

            # magnitude_spectrum[np.isneginf(magnitude_spectrum)] = 0

            return dft_shift, spec

        @staticmethod
        def IFFT(fft_img):
            """Apply an inverse fourier transform

            :Parameters: image_fft
            :Returns: image
            """
            back_ishift = np.fft.ifftshift(fft_img)
            img_back = np.fft.ifft2(back_ishift, axes=(0, 1))
            img_back = np.abs(img_back).clip(0, 255).astype(np.uint8)

            return img_back

        @staticmethod
        def FD_bandpass_filter(img, D0=5, w=10, bptype=0):
            if D0 < 1:
                D0 = 1

            gray = Image.Convert.toGray(img)
            gray_fft = np.fft.fft2(gray)
            gray_fftshift = np.fft.fftshift(gray_fft)
            if bptype == 1:
                kernel = Image.FilterKernels.gaussian_bandpass_kernel(gray, D0, w)
            elif bptype == 2:
                kernel = Image.FilterKernels.butterworth_bandpass_kernel(gray, D0, w)
            else:
                kernel = Image.FilterKernels.ideal_bandpass_kernel(gray, D0, w)
            dst_filtered = np.multiply(kernel, gray_fftshift)
            dst_ifftshift = np.fft.ifftshift(dst_filtered)
            dst_ifft = np.fft.ifft2(dst_ifftshift)
            dst = np.abs(np.real(dst_ifft))
            dst = np.clip(dst, 0, 255)
            out = np.uint8(dst)
            return out, kernel

        def pencilsketch(img):
            """Apply a pencil sketch filter to a grayscale image

            :Parameters: image
            :Returns: image
            """

            def dodgeV2(image, mask):
                return cv.divide(image, 255 - mask, scale=256)

            def burnV2(image, mask):
                return 255 - cv.divide(255 - image, 255 - mask, scale=256)

            img_gray_inv = 255 - img
            img_blur = cv.GaussianBlur(img_gray_inv, ksize=(21, 21),
                                       sigmaX=0, sigmaY=0)
            out = dodgeV2(img, img_blur)

            return out

        def sepia(img):
            """Apply sepia filter

            :Parameters: image
            :Returns: image
            """
            res = img.copy()
            res = cv.cvtColor(res, cv.COLOR_BGR2RGB)  # converting to RGB as sepia matrix is for RGB
            res = np.array(res, dtype=np.float64)
            res = cv.transform(res, np.matrix([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]))

            res[np.where(res > 255)] = 255  # clipping values greater than 255 to 255
            res = np.array(res, dtype=np.uint8)
            res = cv.cvtColor(res, cv.COLOR_RGB2BGR)
            return res

        @staticmethod
        def gaussian_noise(img, prob=0.25):
            """ Add gaussian noise

            :Parameters: image, sigma=0.25
            :Returns: image
            """
            noise_img = img.astype(np.cfloat)
            stddev = prob * 100.0
            noise = np.random.randn(*img.shape) * stddev
            noise_img += noise
            noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
            return noise_img

        def salt_and_pepper_noise(img, prob=0.01):
            """Add salt and pepper noise to an image

            :Parameters:
                - image: numpy array of shape (height, width, channels)
                - prob: probability of adding noise (default is 0.01)

            :Returns: image with added noise
            """
            out = img.copy()

            black, white = np.array([0, 0, 0], dtype='uint8'), np.array([255, 255, 255], dtype='uint8')
            probs = np.random.random(out.shape[:2])
            out[probs < (prob / 2)] = black
            out[probs > 1 - (prob / 2)] = white
            return out

        @staticmethod
        def poisson_noise(img, prob=0.25):
            """ Induce poisson noise

            :Parameters: image, lambda=0.25
            :Returns: image
            """
            # Noise range from 0 to 100
            """
            seed = 42
            data = np.float32(img / 255) #convert to float to add poisson noise
            np.random.seed(seed=seed)
            out = np.random.poisson(data * 256) / 256.
            out = np.uint8(out*255)
            out = np.clip(out, 0, 255).astype(np.uint8) #convert back to UINT8
            """
            # data = np.float32(img) #convert to float to add poisson noise

            data = img.astype(float)

            noise = prob
            # peak = 256.0-noise*(256-32)
            peak = 256.0 - noise * (256)
            # print(noise,peak)
            noise_image = np.random.poisson(data / 255.0 * peak) / peak * 255
            out = np.clip(noise_image, 0, 255).astype(np.uint8)
            return out

        @staticmethod
        def k_means(image, k=3):
            """ k_means clustering

            :Parameters: image, k=3
            :Returns: image
            """

            pixel_vals = image.reshape((-1, 3))
            pixel_vals = np.float32(pixel_vals)
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)
            retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            segmented_data = centers[labels.flatten()]
            segmented_image = segmented_data.reshape((image.shape))
            return segmented_image

        @staticmethod
        def merge2channels(img0, img1):
            """Merge 2 images using 2 colors

            :Parameters: image1, image2
            :Returns: image
            """
            img0 = Image.Convert.toGray(img0)
            img1 = Image.Convert.toGray(img1)

            img0 = Image.Adjust.histostretch_clahe(img0)
            img1 = Image.Adjust.histostretch_clahe(img1)

            img0 = cv.cvtColor(img0, cv.COLOR_GRAY2BGR)
            img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)

            r0, g0, b0 = cv.split(img0)
            r1, g1, b1 = cv.split(img1)
            img3 = cv.merge([b1, g1, r0])
            return img3

        @staticmethod
        def merge3channels(img0, img1, img2):
            """Merge 3 images using 3 colors

            :Parameters: image1, image2, image3
            :Returns: image
            """
            img0 = Image.Adjust.histostretch_clahe(img0)
            img1 = Image.Adjust.histostretch_clahe(img1)
            img2 = Image.Adjust.histostretch_clahe(img2)

            img0 = cv.cvtColor(img0, cv.COLOR_GRAY2BGR)
            img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
            img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

            r0, g0, b0 = cv.split(img0)
            r1, g1, b1 = cv.split(img1)
            r2, g2, b2 = cv.split(img2)
            img3 = cv.merge([b2, g1, r0])
            return img3

        @staticmethod
        def gradient_removal(img, filtersize=513, sigmaX=32):
            """Remove Image gradient
            :Parameters: image, filtersize, sigmaX
            :Returns: image
            """

            gray = Image.Convert.toGray(img)
            gaussianImg = cv.GaussianBlur(gray, (filtersize, filtersize), sigmaX=sigmaX)
            img = (gray - gaussianImg)
            img = Image.Adjust.invert(img)

            return img

        @staticmethod
        def replace_black_pixels_by_median(img):
            """Replace black pixels by the median of neighbours

            :Parameters: image
            :Returns: image
            """

            # Make mask of black pixels, True where black
            # Define a threshold to consider a pixel as black
            black_threshold = 20

            # Create a binary mask of black pixels, where True means black
            blackMask = np.all(img < black_threshold, axis=-1)

            # blackMask = np.all(img == 0, axis=-1)
            median = cv.medianBlur(img, 3)
            res = np.where(blackMask[..., None], median, img)
            return res

        def replace_black_pixels_using_nonlocalmeans(img):
            """Replace black pixels by nonlocalmeans filtering

            :Parameters: image
            :Returns: image
            """
            # Define a threshold to consider a pixel as black
            black_threshold = 20

            # Create a binary mask of black pixels, where True means black
            black_mask = np.all(img < black_threshold, axis=-1)

            # Create a copy of the image to modify only black pixels
            img_copy = img.copy()

            # Apply Gaussian blur to black pixels
            blurred = cv.GaussianBlur(img_copy, (5, 5), 0)
            blurred[~black_mask] = img[~black_mask]

            # Apply non-local means denoising to black pixels
            denoised = cv.fastNlMeansDenoisingColored(blurred, None, 10, 10, 7, 21)
            denoised[~black_mask] = img[~black_mask]

            # Apply morphological operations to black pixels
            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv.morphologyEx(denoised, cv.MORPH_CLOSE, kernel)
            cleaned[~black_mask] = img[~black_mask]
            return cleaned

        @staticmethod
        def remove_islands_colour(img, kernel=13):
            """Remove islands in a colour image using a kernel

            :Parameters: image, kernel=13
            :Returns: image
            """
            # Convert the image to grayscale
            gray = Image.Convert.toGray(img)

            blur = cv.GaussianBlur(gray, (kernel, kernel), 0)
            thresh = cv.threshold(blur, 100, 255, cv.THRESH_BINARY)[1]
            # Remove the noisy regions by applying the mask on the image
            result = cv.bitwise_and(img, img, mask=thresh)
            return result

    class Colormap:

        @staticmethod
        def replace_color_in_colormap(image, colormap_target=cv.COLORMAP_JET, source_color=(0, 0, 0)):
            """Replace a color in an image with one colormap by the color in a target colormap

            :Parameters:
            - image: numpy.ndarray
                The input image in BGR format.
            - source_color: tuple, optional
                The RGB color value to be replaced. Default is (0, 0, 0).

            :Returns:
            - numpy.ndarray
                The modified image.
            """
            target_bgr_color = \
                cv.applyColorMap(np.array([[source_color[0]]], dtype=np.uint8), colormap_target)[0][0]

            # Find pixels with the target grayscale color and replace them with the colormap value
            mask = np.all(image == source_color[0], axis=-1)
            image[mask] = target_bgr_color

            return image

        @staticmethod
        def colormap_jet(img):
            """False color jet

            :Parameters: image
            :Returns: image
            """
            im_color = cv.applyColorMap(img, cv.COLORMAP_JET)
            return im_color

        @staticmethod
        def colormap_hot(img):
            """False color rainbow

            :Parameters: image
            :Returns: image
            """
            im_color = cv.applyColorMap(img, cv.COLORMAP_HOT)
            return im_color

        def matplotlib_to_opencv_colormap(cmap):
            """Convert a Matplotlib colormap to an OpenCV colormap.

            :param cmap: Matplotlib colormap object
            :return: OpenCV colormap
            """
            colormap = np.zeros((256, 1, 3), dtype=np.uint8)
            for i in range(256):
                color = np.array(cmap(i / 255.0)[:3]) * 255
                colormap[i, 0, :] = color[::-1]  # Swap RGB to BGR
            return colormap

        def colormap_tab20(img):
            cmap = plt.get_cmap('tab20')
            colormap = Image.Colormap.matplotlib_to_opencv_colormap(cmap)

            # Apply custom colormap through LUT
            im_color = cv.LUT(img, colormap)
            return im_color

        def colormap_tab20b(img):
            cmap = plt.get_cmap('tab20b')
            colormap = Image.Colormap.matplotlib_to_opencv_colormap(cmap)

            # Apply custom colormap through LUT
            im_color = cv.LUT(img, colormap)
            return im_color

        @staticmethod
        def grayscale_to_color(img, color):
            """ colorize a grayscale image by a given color
            :Parameters: image, color
            :Returns: image
            """

            # Add a small value to grayscale image
            gray_img = img.copy()
            # gray_img += 1

            # Normalize grayscale image to the range [0, 255]
            # gray_img = cv.normalize(gray_img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

            # Create color image
            color_img = np.zeros((gray_img.shape[0], gray_img.shape[1], 3), dtype=np.uint8)
            color_img[:, :, 0] = color[0]
            color_img[:, :, 1] = color[1]
            color_img[:, :, 2] = color[2]
            color_img = color_img.astype(np.uint8) * gray_img[:, :, np.newaxis].astype(np.uint8)

            return color_img

        @staticmethod
        def RGBtoLAB(source, target):
            """ convert RGB to LAB color space

            :Parameters: source_image, target_image
            :Returns: image
            """
            # convert the images from the RGB to L*ab* color space, being
            # sure to utilizing the floating point data type (note: OpenCV
            # expects floats to be 32-bit, so use that instead of 64-bit)
            source = cv.cvtColor(source, cv.COLOR_GRAY2BGR)
            target = cv.cvtColor(target, cv.COLOR_GRAY2BGR)

            source = cv.cvtColor(source, cv.COLOR_BGR2LAB).astype("float32")
            target = cv.cvtColor(target, cv.COLOR_BGR2LAB).astype("float32")

            # compute color statistics for the source and target images
            (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = _image_stats(source)
            (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = _image_stats(target)

            # subtract the means from the target image
            (l, a, b) = cv.split(target)
            l -= lMeanTar
            a -= aMeanTar
            b -= bMeanTar

            # scale by the standard deviations
            l = (lStdTar / lStdSrc) * l
            a = (aStdTar / aStdSrc) * a
            b = (bStdTar / bStdSrc) * b

            # add in the source mean
            l += lMeanSrc
            a += aMeanSrc
            b += bMeanSrc

            # clip the pixel intensities to [0, 255] if they fall outside
            # this range
            l = np.clip(l, 0, 255)
            a = np.clip(a, 0, 255)
            b = np.clip(b, 0, 255)

            # merge the channels together and convert back to the RGB color
            # space, being sure to utilize the 8-bit unsigned integer data
            # type
            transfer = cv.merge([l, a, b])
            transfer = cv.cvtColor(transfer.astype("uint8"), cv.COLOR_LAB2BGR)

            # return the color transferred image
            return transfer

    class Adjust:
        @staticmethod
        def invert(img):
            """Invert image

            :Parameters: image
            :Returns: image
            """
            img2 = cv.bitwise_not(img)
            return img2

        @staticmethod
        def squared_and_bin(img):
            """First make image squared followed by binning to 256 pixels

            :Parameters: image
            :Returns: image
            """
            img0 = Image.Tools.squared(img, leadingaxislargest=False)
            scale = 256 / img0.shape[1]
            img0 = cv.resize(img0, None, None, scale, scale, interpolation=cv.INTER_AREA)
            return img0

        @staticmethod
        def bin(img, shrinkfactor=2):
            """bin image with shrinkfactor (default shrinkfactor= 2)
            :Parameters: image, shrinkfactor
            :Returns: image
            """
            scale = 1 / shrinkfactor
            img0 = cv.resize(img, None, None, scale, scale, interpolation=cv.INTER_AREA)
            return img0

        @staticmethod
        def histogram(img):
            """create histogram of an image as an image

            :Parameters: image
            :Output: histogram image
            """

            w = img.shape[1]
            h = img.shape[0]
            if (img.dtype == np.uint8):
                rng = 256
            else:
                rng = 65535
            # bitdepth = img.dtype
            hist, bins = np.histogram(img.flatten(), 256, [0, rng])
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max()  # this line not necessary.

            fig = plt.figure()

            plt.plot(cdf_normalized, color='b')
            plt.hist(img.flatten(), 256, [0, rng], color='0.30')

            plt.axis("off")  # turns off axes
            fig.tight_layout()

            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            out = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            # cv.imwrite("test.png",out)
            return out

        @staticmethod
        def histostretch_clahe(img):
            """Apply a CLAHE (Contrast Limited Adaptive Histogram Equalization) filter on a grayscale image
            supports 8 and 16 bit images.

            :Parameters: image
            :Returns: image
            """
            # img = cv.pyrDown(img)

            if (len(img.shape) < 3):
                clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl1 = clahe.apply(img)
                img = cl1
            else:
                clahe = cv.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
                lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)  # convert from BGR to LAB color space
                l, a, b = cv.split(lab)  # split on 3 different channels
                l2 = clahe.apply(l)  # apply CLAHE to the L-channel
                lab = cv.merge((l2, a, b))  # merge channels
                img = cv.cvtColor(lab, cv.COLOR_LAB2BGR)  # convert from LAB to BGR
            return img

        @staticmethod
        def histostretch_equalized(img):
            """Apply a equalize histogram filter
            8 and 16 bit

            :Parameters: image
            :Returns: image

            #https://github.com/torywalker/histogram-equalizer/blob/master/HistogramEqualization.ipynb
            """

            def get_histogram(image, bins):
                # array with size of bins, set to zeros
                histogram = np.zeros(bins)

                # loop through pixels and sum up counts of pixels
                for pixel in image:
                    histogram[pixel] += 1

                # return our final result
                return histogram

            # create our cumulative sum function
            def cumsum(a):
                a = iter(a)
                b = [next(a)]
                for i in a:
                    b.append(b[-1] + i)
                return np.array(b)

            if (img.dtype == np.uint16):
                flat = img.flatten()
                hist = get_histogram(flat, 65536)
                # plt.plot(hist)
                #
                cs = cumsum(hist)
                # re-normalize cumsum values to be between 0-255

                # numerator & denomenator
                nj = (cs - cs.min()) * 65535
                N = cs.max() - cs.min()

                # re-normalize the cdf
                cs = nj / N
                cs = cs.astype('uint16')
                img_new = cs[flat]
                # plt.hist(img_new, bins=65536)
                # plt.show(block=True)
                img_new = np.reshape(img_new, img.shape)
            else:
                if len(img.shape) == 2:
                    img_new = cv.equalizeHist(img)
                else:
                    img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)  # equalize the histogram of the Y channel
                    img_yuv[:, :, 0] = cv.equalizeHist(img_yuv[:, :, 0])  # convert the YUV image back to RGB format
                    img_new = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)

            return img_new

        @staticmethod
        def histostretch_normalize(img):
            """Normalize histogram
            8bit between 0 and 255
            16bit between 0 and 65535

            :Parameters: image
            :Returns: image
            """
            # img = cv.pyrDown(img)
            if (img.dtype == np.uint16):
                norm = cv.normalize(img, None, alpha=0, beta=65535, norm_type=cv.NORM_MINMAX, dtype=cv.CV_16U)
            else:
                norm = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
            return norm

        # smooth, threshold
        @staticmethod
        def threshold(img, thresh=128):
            """Applies a fixed-level threshold to each array element. [0-255]

            :Parameters: image, threshold
            :Returns: image
            """
            ret, imout = cv.threshold(img, thresh, 255, cv.THRESH_BINARY)
            return imout

        @staticmethod
        def normalize(img):
            """Normalize image. [0-255]

            :Parameters: image
            :Returns: image
            """
            imout = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)
            return imout

        @staticmethod
        def thresholdrange(img, threshmin=128, threshmax=255):
            """threshold image between min and max value

            :Parameters: image, thresholdmin, thresholdmax
            :Returns: image
            """
            imout = cv.inRange(img, threshmin, threshmax)
            return imout

        @staticmethod
        def threshold_otsu(img):
            """Applies an automatic threshold using the Otsu method for thresholding

            :Parameters: image
            :Returns: image
            """
            ret, imout = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
            return imout

        @staticmethod
        def adjust_contrast_brightness(img, contrast=0, brightness=0):
            """adjust contrast and brightness
                contrast range: -127..127
                brightness range: -255..255

            :Parameters: image
            :Returns: image
            """
            table = np.array([i * (contrast / 127 + 1) - contrast + brightness for i in range(0, 256)]).clip(0,
                                                                                                             255).astype(
                'uint8')
            # if len(img.shape) == 3:
            # out = cv.LUT(img, table)[:, :, np.newaxis]
            # else:
            out = cv.LUT(img, table)
            return out

        @staticmethod
        def adjust_gamma(image, gamma=1.0):
            """adjust gamma [0..3.0], default = 1
            gamma cannot be 0

            :Parameters: image, gamma=1.0
            :Returns: image
            """
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255
                              for i in np.arange(0, 256)]).astype("uint8")

            # apply gamma correction using the lookup table
            return cv.LUT(image, table)

        @staticmethod
        def adjust_HSV(img, hval, sval, vval):
            """adjust Hue [0..179], Saturation [-255..255], lightness [-255..255]

            :Parameters: image, hue, saturation, lightness
            :Returns: image
            """
            img = Image.Convert.toRGB(img)  # changing channels for nicer image
            hsv = Image.Convert.BGRtoHSV(img)

            h = hsv[:, :, 0]
            s = hsv[:, :, 1]
            v = hsv[:, :, 2]

            h = np.where(h <= 255.0 - hval, h + hval, 255)
            if (sval > 0):
                s = np.where(s <= 255.0 - sval, s + sval, 255)
            else:
                s = (s * ((255.0 + sval) / 255.0))
            if (vval > 0):
                v = np.where(v <= 255.0 - vval, v + vval, 255)
            else:
                v = v * ((255.0 + vval) / 255.0)

            hsv[:, :, 0] = h
            hsv[:, :, 1] = s
            hsv[:, :, 2] = v

            img1 = Image.Convert.HSVtoBGR(hsv)
            return img1

        @staticmethod
        def adjust_HSL(img, hval, sval, lval):
            """adjust Hue [0..179], Saturation [0..255], lightness [0..255]

            The definition HSL is most commonly used, occasionly this is called HLS

            :Parameters: image, hue, saturation, lightness
            :Returns: image
            """

            img = Image.Convert.toRGB(img)  # changing channels for nicer image

            hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
            h = hls[:, :, 0]
            l = hls[:, :, 1]
            s = hls[:, :, 2]

            h = np.where(h <= 255.0 - hval, h + hval, 255)
            if (sval > 0):
                s = np.where(s <= 255.0 - sval, s + sval, 255)
            else:
                s = (s * ((255.0 + sval) / 255.0))
            if (lval > 0):
                l = np.where(l <= 255.0 - lval, l + lval, 255)
            else:
                l = l * ((255.0 + lval) / 255.0)

            hls[:, :, 0] = h
            hls[:, :, 1] = l
            hls[:, :, 2] = s

            img1 = cv.cvtColor(hls, cv.COLOR_HLS2RGB)
            return img1

        @staticmethod
        def adjust_auto_whitebalance(img):
            """auto whitebalance

            https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
            :Parameters: image, temperature
            :Returns: image
            """

            result = cv.cvtColor(img, cv.COLOR_BGR2LAB)
            avg_a = np.average(result[:, :, 1])
            avg_b = np.average(result[:, :, 2])
            result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
            result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
            result = cv.cvtColor(result, cv.COLOR_LAB2BGR)
            return result

    class Transform:
        @staticmethod
        def flip_horizontal(img):
            """Flip image horizontal

            :Parameters: image
            :Returns: image
            """
            horizontal_img = cv.flip(img, 0)
            return horizontal_img

        @staticmethod
        def flip_vertical(img):
            """Flip image vertical

            :Parameters: image
            :Returns: image
            """
            vertical_img = cv.flip(img, 1)
            return vertical_img

        @staticmethod
        def translate(img, shiftx, shifty):
            """Shift image n x and y pixels

            :Parameters: image, shiftx, shifty
            :Returns: image
            """
            w = img.shape[1]
            h = img.shape[0]

            M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
            img2 = cv.warpAffine(img, M, (w, h))
            return img2

        @staticmethod
        def rotate(image, angle):
            """Rotate image

            :Parameters: image, angle
            :Returns: image
            """
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
            result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
            return result

    class Binary:

        @staticmethod
        def skeletonize(img):
            """skeletonize a thresholded image.

            :Parameters: image
            :Returns: image
            """
            size = np.size(img)
            skel = np.zeros(img.shape, np.uint8)
            element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
            done = False

            while (not done):
                eroded = cv.erode(img, element)
                temp = cv.dilate(eroded, element)
                temp = cv.subtract(img, temp)
                skel = cv.bitwise_or(skel, temp)
                img = eroded.copy()

                zeros = size - cv.countNonZero(img)
                if zeros == size:
                    done = True
            return skel

        # Zhang-Suen Thinning Algorithm - https://github.com/linbojin/Skeletonization-by-Zhang-Suen-Thinning-Algorithm
        # note: slow filter
        @staticmethod
        def zhang_suen_thinning(img):
            """Applies the Zhang-Suen thinning algorithm.

            :Parameters: image
            :Returns: image
            """

            def neighbours(x, y, img):
                "Return 8-neighbours of image point P1(x,y), in a clockwise order"
                img = img
                x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
                return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],  # P2,P3,P4,P5
                        img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1]]  # P6,P7,P8,P9

            def transitions(neighbours):
                "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
                n = neighbours + neighbours[0:1]  # P2, P3, ... , P8, P9, P2
                return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

            ret, imout = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
            img = img < ret  # must set object region as 1, background region as 0 !

            print("the Zhang-Suen Thinning Algorithm")
            img_Thinned = img.copy()  # deepcopy to protect the original img
            changing1 = changing2 = 1  # the points to be removed (set as 0)
            while changing1 or changing2:  # iterates until no further changes occur in the img
                # Step 1
                changing1 = []
                rows, columns = img_Thinned.shape  # x for rows, y for columns
                for x in range(1, rows - 1):  # No. of  rows
                    for y in range(1, columns - 1):  # No. of columns
                        P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, img_Thinned)
                        if (img_Thinned[x][y] == 1 and  # Condition 0: Point P1 in the object regions
                                2 <= sum(n) <= 6 and  # Condition 1: 2<= N(P1) <= 6
                                transitions(n) == 1 and  # Condition 2: S(P1)=1
                                P2 * P4 * P6 == 0 and  # Condition 3
                                P4 * P6 * P8 == 0):  # Condition 4
                            changing1.append((x, y))
                for x, y in changing1:
                    img_Thinned[x][y] = 0
                # Step 2
                changing2 = []
                for x in range(1, rows - 1):
                    for y in range(1, columns - 1):
                        P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, img_Thinned)
                        if (img_Thinned[x][y] == 1 and  # Condition 0
                                2 <= sum(n) <= 6 and  # Condition 1
                                transitions(n) == 1 and  # Condition 2
                                P2 * P4 * P8 == 0 and  # Condition 3
                                P2 * P6 * P8 == 0):  # Condition 4
                            changing2.append((x, y))
                for x, y in changing2:
                    img_Thinned[x][y] = 0
            return img_Thinned

        @staticmethod
        def morphology_erode(img, kernel=5):
            """Morphology filter - erode

            :Parameters: image, kernel
            :Returns: image
            """
            kerneln = np.ones((kernel, kernel), np.uint8)
            erosion = cv.erode(img, kerneln, iterations=1)
            return erosion

        @staticmethod
        def morphology_dilate(img, kernel=5):
            """Morphology filter - dilate

            :Parameters: image, kernel
            :Returns: image
            """
            kerneln = np.ones((kernel, kernel), np.uint8)
            dilation = cv.dilate(img, kerneln, iterations=1)
            return dilation

        @staticmethod
        def morphology_open(img, kernel=5):
            """Morphology filter - open

            :Parameters: image, kernel
            :Returns: image
            """
            kerneln = np.ones((kernel, kernel), np.uint8)
            opening = cv.morphologyEx(img, cv.MORPH_OPEN, kerneln)
            return opening

        @staticmethod
        def morphology_close(img, kernel=5):
            """Morphology filter - close

            :Parameters: image, kernel
            :Returns: image
            """
            kerneln = np.ones((kernel, kernel), np.uint8)
            opening = cv.morphologyEx(img, cv.MORPH_CLOSE, kerneln)
            return opening

        @staticmethod
        def morphology_fillholes(im_in):
            """Morphology filter - fillholes

            :Parameters: image, kernel
            :Returns: image
            """
            im_floodfill = im_in.copy()
            # Mask used to flood filling.
            # Notice the size needs to be 2 pixels than the image.
            h, w = im_in.shape[:2]
            mask = np.zeros((h + 2, w + 2), np.uint8)
            # Floodfill from point (0, 0)
            cv.floodFill(im_floodfill, mask, (0, 0), 255)
            # Invert floodfilled image
            im_floodfill_inv = cv.bitwise_not(im_floodfill)
            # Combine the two images to get the foreground.
            im_out = im_in | im_floodfill_inv

            return im_in, im_floodfill, im_floodfill_inv, im_out

        @staticmethod
        def remove_isolated_pixels(img0):
            """Remove isolated pixels in an image

            :Parameters: image
            :Returns: image
            """
            input_image = cv.threshold(img0, 254, 255, cv.THRESH_BINARY)[1]
            input_image_comp = cv.bitwise_not(input_image)  # could just use 255-img

            kernel1 = np.array([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]], np.uint8)
            kernel2 = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]], np.uint8)

            hitormiss1 = cv.morphologyEx(input_image, cv.MORPH_ERODE, kernel1)
            hitormiss2 = cv.morphologyEx(input_image_comp, cv.MORPH_ERODE, kernel2)
            hitormiss = cv.bitwise_and(hitormiss1, hitormiss2)
            hitormiss_comp = cv.bitwise_not(hitormiss)  # could just use 255-img
            del_isolated = cv.bitwise_and(input_image, input_image, mask=hitormiss_comp)
            return del_isolated

        @staticmethod
        def remove_islands(img0, min_size=150):
            """Remove islands in an image

            :Parameters: image, min_size=150
            :Returns: image
            """

            # find all your connected components (white blobs in your image)
            nb_components, output, stats, centroids = cv.connectedComponentsWithStats(img0, connectivity=8)
            # connectedComponentswithStats yields every seperated component with information on each of them, such as size
            # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
            sizes = stats[1:, -1]
            nb_components = nb_components - 1

            # minimum size of features we want to keep (number of pixels)
            # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever

            # your answer image
            img2 = np.zeros((output.shape))
            # for every component in the image, you keep it only if it's above min_size
            for i in range(0, nb_components):
                if sizes[i] >= min_size:
                    img2[output == i + 1] = 255
            return img2

    class Convert:

        @staticmethod
        def to8bit(img):
            """Convert to 8 bit image

            :Parameters: image
            :Returns: image
            """
            if (img.dtype == np.uint16):
                img1 = (img / 256).astype('uint8')  # updated this one on 20191216 for 16 bit imaging
            else:
                img1 = (img).astype('uint8')
            # img1 = img.astype('uint8')  # 16bit to 8bit
            return img1

        @staticmethod
        def to16bit(img):
            """Convert to 16 bit image

            :Parameters: image
            :Returns: image
            """
            if (img.dtype == np.uint8):
                img1 = (img * 256).astype('uint16')  # updated this one on 20191216 for 16 bit imaging
            else:
                img1 = (img).astype('uint16')
            # img1 = img.astype('uint8')  # 16bit to 8bit
            return img1

        @staticmethod
        def toRGB(img):
            """Convert grayscale to RGB image

            :Parameters: image
            :Returns: image
            """
            img1 = img
            channels = len(img.shape)
            if (channels != 3):
                img1 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
                # print('Image converted from Grayscale to RGB')
            if (channels == 3 and img.shape[2] == 4):
                img1 = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
                # remove alpha channel
            return img1

        @staticmethod
        def toGray(img):
            """Convert RGB to color grayscale image

            :Parameters: image
            :Returns: image
            """
            img1 = img
            channels = len(img.shape)
            if (channels > 2):
                img1 = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
                # print('Image converted from RGB to Grayscale')
            return img1

        @staticmethod
        def BGRtoRGB(img):
            """Convert BGR to RGB

            :Parameters: image
            :Returns: image
            """
            img1 = img
            channels = len(img.shape)
            if (channels == 3 and img.shape[2] == 4):
                img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
                # remove alpha channel
            if (channels > 2):
                b, g, r = cv.split(img)  # get b,g,r
                img1 = cv.merge([r, g, b])  # switch it to rgb (OpenCV uses BGR)
            return img1

        @staticmethod
        def RGBtoBGR(img):
            """Convert RGB to BGR

            :Parameters: image
            :Returns: image
            """
            img1 = img
            channels = len(img.shape)
            if (channels == 3 and img.shape[2] == 4):
                img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
                # remove alpha channel
            if (channels > 2):
                r, g, b = cv.split(img)  # get b,g,r
                img1 = cv.merge([b, g, r])  # switch it to rgb (OpenCV uses BGR)
            return img1

        @staticmethod
        def BGRtoHSV(img):
            """Convert BGR to HSV

            :Parameters: image
            :Returns: image
            """
            img1 = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            return img1

        @staticmethod
        def HSVtoBGR(img):
            """Convert HSV to BGR

            :Parameters: image
            :Returns: image
            """
            img1 = cv.cvtColor(img, cv.COLOR_HSV2BGR)
            return img1

        @staticmethod
        def binarytogray(img):
            """Convert binary image to grayscale (dtype=bool -> dtype=uint8)

            :Parameters: image
            :Returns: image
            """
            img = img.astype('uint8') * 255
            return img

    class FilterKernels:

        @staticmethod
        def ideal_lowpass_kernel(img, radius=32):
            rows, cols = img.shape[:2]
            r, c = np.mgrid[0:rows:1, 0:cols:1]
            c -= int(cols / 2)
            r -= int(rows / 2)
            d = np.power(c, 2.0) + np.power(r, 2.0)
            kernel_matrix = np.zeros((rows, cols), np.float32)
            kernel = np.copy(d)
            kernel[kernel < pow(radius, 2.0)] = 1
            kernel[kernel >= pow(radius, 2.0)] = 0
            kernel_matrix[:, :] = kernel
            return kernel_matrix

        @staticmethod
        def gaussian_lowpass_kernel(img, radius=32):
            rows, cols = img.shape[:2]
            r, c = np.mgrid[0:rows:1, 0:cols:1]
            c -= int(cols / 2)
            r -= int(rows / 2)
            d = np.power(c, 2.0) + np.power(r, 2.0)
            kernel_matrix = np.zeros((rows, cols), np.float32)
            kernel = np.exp(-d / (2 * pow(radius, 2.0)))
            kernel_matrix[:, :] = kernel
            return kernel_matrix

        @staticmethod
        def butterworth_lowpass_kernel(img, radius=32, n=2):
            rows, cols = img.shape[:2]
            r, c = np.mgrid[0:rows:1, 0:cols:1]
            c -= int(cols / 2)
            r -= int(rows / 2)
            d = np.power(c, 2.0) + np.power(r, 2.0)
            kernel_matrix = np.zeros((rows, cols), np.float32)
            kernel = 1.0 / (1 + np.power(np.sqrt(d) / radius, 2 * n))
            kernel_matrix[:, :] = kernel
            return kernel_matrix

        @staticmethod
        def ideal_bandpass_kernel2(img, D0=32, w=9):
            rows, cols = img.shape[:2]
            crow, ccol = int(rows / 2), int(cols / 2)
            mask = np.ones((rows, cols), np.uint8)
            for i in range(0, rows):
                for j in range(0, cols):
                    d = np.sqrt(pow(i - crow, 2) + pow(j - ccol, 2))
                    if D0 - w / 2 < d < D0 + w / 2:
                        mask[i, j] = 1
                    else:
                        mask[i, j] = 0
            kernel = mask
            return kernel

        def ideal_bandpass_kernel(img, D0=32, w=9):
            rows, cols = img.shape[:2]
            crow, ccol = rows // 2, cols // 2

            # Create a meshgrid of row and column indices
            rows_idx, cols_idx = np.ogrid[:rows, :cols]

            # Calculate the distance from the center of the image
            distance = np.sqrt((rows_idx - crow) ** 2 + (cols_idx - ccol) ** 2)

            # Create a mask with ones within the specified distance from the center
            mask = np.zeros((rows, cols), np.uint8)
            mask[(distance >= D0 - w / 2) & (distance <= D0 + w / 2)] = 1

            resized_kernel = cv.resize(mask, (img.shape[1], img.shape[0]))

            # Multiply the mask with the bandpass kernel to create the final kernel
            kernel = mask * resized_kernel
            return kernel

        @staticmethod
        def ideal_bandstop_kernel(img, D0=32, W=9):
            kernel = 1.0 - Image.FilterKernels.ideal_bandpass_kernel(img, D0, W)
            return kernel

        @staticmethod
        def gaussian_bandstop_kernel(img, D0=32, W=9):
            r, c = img.shape[1], img.shape[0]
            u = np.arange(r)
            v = np.arange(c)
            u, v = np.meshgrid(u, v)
            low_pass = np.sqrt((u - r / 2) ** 2 + (v - c / 2) ** 2)
            kernel = 1.0 - np.exp(-0.5 * (((low_pass ** 2 - D0 ** 2) / (low_pass * W + 1.0e-5)) ** 2))
            return kernel

        @staticmethod
        def gaussian_bandpass_kernel(img, D0=32, W=9):
            assert img.ndim == 2
            # kernel = Image.FilterKernels.gaussian_bandstop_kernel(img, D0, W)
            kernel = 1.0 - Image.FilterKernels.gaussian_bandstop_kernel(img, D0, W)
            return kernel

        @staticmethod
        def butterworth_bandstop_kernel(img, D0=32, W=9, n=1):
            r, c = img.shape[1], img.shape[0]
            u = np.arange(r)
            v = np.arange(c)
            u, v = np.meshgrid(u, v)
            low_pass = np.sqrt((u - r / 2) ** 2 + (v - c / 2) ** 2)
            kernel = (1 / (1 + ((low_pass * W) / (low_pass ** 2 - D0 ** 2)) ** (2 * n)))
            return kernel

        def butterworth_bandpass_kernel(img, D0=5, W=10):
            kernel = 1.0 - Image.FilterKernels.butterworth_bandstop_kernel(img, D0, W)
            return kernel

    class Tools:
        # combined sequences

        @staticmethod
        def image_with_2_closeups(img, t_size=[0.2, 0.2], t_center1=[0.3, 0.3], t_center2=[0.6, 0.6]):
            """image with 2 closeups, the output is a color image.

            :Parameters: image, t_size=[0.2, 0.2], t_center1=[0.3, 0.3], t_center2=[0.6, 0.6]
            :Returns: image
            """

            w = img.shape[1]
            h = img.shape[0]

            rgb = Image.Convert.toRGB(img)
            xt0 = Image._multipleof2((t_center1[0] - t_size[0] * 0.5) * w)
            yt0 = Image._multipleof2((t_center1[1] - t_size[1] * 0.5) * h)
            xt1 = Image._multipleof2((t_center1[0] + t_size[0] * 0.5) * w)
            yt1 = Image._multipleof2((t_center1[1] + t_size[1] * 0.5) * h)
            # rgb = img
            template1 = Image.crop(rgb, xt0, yt0, xt1, yt1)
            w3 = np.abs(xt0 - xt1)
            h3 = np.abs(yt0 - yt1)

            xt0b = Image._multipleof2((t_center2[0] - t_size[0] * 0.5) * w)
            yt0b = Image._multipleof2((t_center2[1] - t_size[1] * 0.5) * h)
            # rgb = img
            template2 = Image.crop(rgb, xt0b, yt0b, xt0b + w3, yt0b + h3)

            wt = template1.shape[1]
            ht = template1.shape[0]
            scalefactor = (w * 0.5) / wt
            template1b = Image.resize(template1, scalefactor)
            # print(template1b.shape)

            wt2 = template1b.shape[1]
            ht2 = template1b.shape[0]
            template2b = Image.resize(template2, scalefactor)
            # print(template2b.shape)

            # print(w,h)
            # print(wt2,ht2)

            output = np.zeros((h + ht2, w, 3), np.uint8)

            print(output.shape)
            print(rgb.shape)
            print(template1b.shape)
            print(template2b.shape)

            output[0:h, 0:w] = rgb
            output[h:h + ht2, 0:wt2] = template1b
            output[h:h + ht2, wt2:w] = template2b

            output = cv.rectangle(output, (xt0, yt0), (xt1, yt1), (33, 145, 237), 3)
            output = cv.rectangle(output, (xt0b, yt0b), (xt0b + w3, yt0b + h3), (240, 167, 41), 3)

            output = cv.rectangle(output, (wt2 + 3, h), (w - 2, h + ht2 - 3), (240, 167, 41), 3)
            output = cv.rectangle(output, (0 + 2, h), (wt2 - 2, h + ht2 - 3), (33, 145, 237), 3)

            return output

        @staticmethod
        def anaglyph(img0, img1):
            """Create a anaglyph from 2 images (stereo image)

            :Parameters: image1, image2
            :Returns: image
            """
            matrices = {
                'true': [[0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0.299, 0.587, 0.114]],
                'mono': [[0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0.299, 0.587, 0.114, 0.299, 0.587, 0.114]],
                'color': [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 1]],
                'halfcolor': [[0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 1]],
                'optimized': [[0, 0.7, 0.3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 1]],
            }

            # img1 = translate_image(img1,8,0)

            width = img0.shape[0]
            height = img0.shape[1]
            leftImage = cv.cvtColor(img0, cv.COLOR_GRAY2BGR)
            rightImage = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)

            m = matrices['optimized']
            result = np.zeros((img0.shape[0], img0.shape[1], 3), np.uint8)

            # split the left and right images into separate blue, green and red images
            lb, lg, lr = cv.split(np.asarray(leftImage[:, :]))
            rb, rg, rr = cv.split(np.asarray(rightImage[:, :]))
            resultArray = np.asarray(result[:, :])
            resultArray[:, :, 0] = lb * m[0][6] + lg * m[0][7] + lr * m[0][8] + rb * m[1][6] + rg * m[1][7] + rr * m[1][
                8]
            resultArray[:, :, 1] = lb * m[0][3] + lg * m[0][4] + lr * m[0][5] + rb * m[1][3] + rg * m[1][4] + rr * m[1][
                5]
            resultArray[:, :, 2] = lb * m[0][0] + lg * m[0][1] + lr * m[0][2] + rb * m[1][0] + rg * m[1][1] + rr * m[1][
                2]
            return result

        @staticmethod
        def image2patches(img, patchsize, overlap_px=0, verbose=False):
            """
            Convert single image to a list of patches.
            The size of a patch is determined by patchsize, be aware of rounding incase image width or height cannot be divided through the patchsize.
            Works both for color and grayscale images.
            overlap in pixels (default overlap=0)

            :Parameters: image, rows, cols
            :Returns: image_list
            """
            h0, w0 = img.shape[0], img.shape[1]
            # determine number of steps (rows and columns
            cols = int(np.round(w0 / patchsize, 0))
            rows = int(np.round(h0 / patchsize, 0))
            if (cols < 1):
                cols = 1
            if (rows < 1):
                rows = 1

            h0_size = int(h0 / rows + 0.5)
            w0_size = int(w0 / cols + 0.5)

            # add black border to image
            bordersize = int(overlap_px)  # require bordersize of the patches

            channels = len(img.shape)
            if (channels == 3):
                # color image
                base_size = h0 + bordersize * 2, w0 + bordersize * 2, 3
                base = np.zeros((base_size), np.uint8)
            else:
                base_size = h0 + bordersize * 2, w0 + bordersize * 2
                base = np.zeros((base_size), np.uint8)

            # base = np.zeros(base_size, dtype=np.uint8)
            base[bordersize:h0 + bordersize, bordersize:w0 + bordersize] = img  # this works

            # make patches with overlap
            patches = []
            for row in range(rows):
                for col in range(cols):
                    yc = int((row + 0.5) * h0_size) + bordersize
                    xc = int((col + 0.5) * w0_size) + bordersize
                    x0 = int(xc - (w0_size * 0.5) - bordersize)
                    y0 = int(yc - (h0_size * 0.5) - bordersize)
                    x1 = int(xc + (w0_size * 0.5) + bordersize)
                    y1 = int(yc + (h0_size * 0.5) + bordersize)
                    patch = base[y0:y1, x0:x1]
                    patches.append(patch)

            if verbose == True:
                print(
                    "image2patches: patches {}, source_width {}, source_height {},rows {}, columns {}, output: patches,cols".format(
                        len(patches), w0, h0, rows, cols))
            return patches, cols

        def patches2image_overlay(images, cols=5, overlap_px=0, whitebackground=True, verbose=False):
            """
            Stitch a list of image patches to a single image. The number of columns determines the next line.
            Works both for color and grayscale images.
            overlap in pixels (default overlap=0)
            Other definitions often used for this process: image montage or image stitching
            when cols is set to 0 rows and cols will be equal.
            with overlay a mask is applied to connect the boundaries.

            :Parameters: imagelist, cols=5, overlap_perc=0, whitebackground=True
            :Returns: image
            """

            if (cols == 0):
                cols = int(np.math.sqrt(len(images)))
                rows = cols
                if verbose == True:
                    print('patches2image equal rows and columns')
            else:
                if (cols > len(images)):
                    cols = len(images)

                rows = int(len(images) / cols + 0.5)
                if (rows * cols) < len(images):
                    cols = cols + (len(images) - (rows * cols))  # number of total images should be correct

            maxwidth = max(image.shape[1] for image in images)
            maxheight = max(image.shape[0] for image in images)
            gap = int(-overlap_px * 2.)
            # maxwidth = maxwidth
            # maxheight = maxheight

            height = maxheight * rows + (gap * (rows - 1))
            width = maxwidth * cols + (gap * (cols - 1))
            # output = np.zeros((height, width), np.uint8)
            if verbose == True:
                print(
                    "patches2image images {}, new_width {}, new_height {}, rows {}, cols {}, gap {}".format(len(images),
                                                                                                            width,
                                                                                                            height,
                                                                                                            rows, cols,
                                                                                                            gap))
            channels = len(images[0].shape)
            if (channels == 3):
                # color image
                output2 = np.zeros((height, width, 3), np.uint8)
            else:
                output2 = np.zeros((height, width), np.uint8)

            x = 0
            y = 0
            for image in images:

                if (channels == 3):
                    # color image
                    output = np.zeros((height, width, 3), np.uint8)
                else:
                    output = np.zeros((height, width), np.uint8)

                if (whitebackground == True):
                    cv.bitwise_not(output, output)

                # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # changing image to grayscale
                h, w = image.shape[0], image.shape[1]

                output[(y * h + gap * y):((y + 1) * h + gap * y), (x * w + gap * x):((x + 1) * w + gap * x)] = image

                output2 = Image.add_overlay(output2, output)
                x += 1
                if (x > (cols - 1)):
                    x = 0

                    y += 1
            # out = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)  # and back
            h4, w4 = output2.shape[0], output.shape[1]
            out = output2[overlap_px:h4 - overlap_px, overlap_px:w4 - overlap_px]
            return out

        def patches2image(images, cols=5, overlap_px=0, whitebackground=True, verbose=False):
            """
            Stitch a list of image patches to a single image. The number of columns determines the next line.
            Works both for color and grayscale images.
            overlap in pixels (default overlap=0)
            Other definitions often used for this process: image montage or image stitching
            when cols is set to 0 rows and cols will be equal.

            :Parameters: imagelist, cols=5, overlap_perc=0, whitebackground=True
            :Returns: image
            """

            if (cols == 0):
                cols = int(np.math.sqrt(len(images)))
                rows = cols
                if verbose == True:
                    print('patches2image equal rows and columns')
            else:
                if (cols > len(images)):
                    cols = len(images)

                rows = int(len(images) / cols + 0.5)
                if (rows * cols) < len(images):
                    cols = cols + (len(images) - (rows * cols))  # number of total images should be correct

            maxwidth = max(image.shape[1] for image in images)
            maxheight = max(image.shape[0] for image in images)
            gap = int(-overlap_px * 2.)
            # maxwidth = maxwidth
            # maxheight = maxheight

            height = maxheight * rows + (gap * (rows - 1))
            width = maxwidth * cols + (gap * (cols - 1))
            # output = np.zeros((height, width), np.uint8)
            if verbose == True:
                print(
                    "patches2image images {}, new_width {}, new_height {}, rows {}, cols {}, gap {}".format(len(images),
                                                                                                            width,
                                                                                                            height,
                                                                                                            rows, cols,
                                                                                                            gap))
            channels = len(images[0].shape)
            if (channels == 3):
                # color image
                output = np.zeros((height, width, 3), np.uint8)
            else:
                output = np.zeros((height, width), np.uint8)

            if (whitebackground == True):
                cv.bitwise_not(output, output)
            x = 0
            y = 0
            for image in images:

                # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # changing image to grayscale
                h, w = image.shape[0], image.shape[1]
                output[(y * h + gap * y):((y + 1) * h + gap * y), (x * w + gap * x):((x + 1) * w + gap * x)] = image
                x += 1
                if (x > (cols - 1)):
                    x = 0

                    y += 1
            # out = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)  # and back
            h4, w4 = output.shape[0], output.shape[1]
            out = output[overlap_px:h4 - overlap_px, overlap_px:w4 - overlap_px]
            return out

        @staticmethod
        def patches2disk(folder, patches):
            """
            Save list of patches to disk

            :Parameters: path patches
            """
            for t in range(0, len(patches)):
                cv.imwrite(os.path.join(folder, "patch_{0}.png".format(t)), patches[t])

        @staticmethod
        def create_hsv_map():
            """
            generate a HSV Map pattern
            :Parameters: -
            :Returns: image
            """

            V, H = np.mgrid[0:1:100j, 0:1:300j]
            S = np.ones_like(V)
            HSV = np.dstack((H, S, V))

            out = hsv_to_rgb(HSV)
            # plt.imshow(out)
            # out = Image.Convert.HSVtoBGR(np.float32(HSV))
            # out = Image.Convert.BGRtoRGB(out)
            return out

        @staticmethod
        def create_checkerboard(rows_num=10, columns_num=10, block_size=30, base_col=(255, 255, 255)):
            """
            generate a checkerboard pattern
            :Parameters: rows, columns, blocksize, base color
            :Returns: image
            """
            base_color = tuple(map(int, base_col))
            block_size = block_size * 4
            image_width = block_size * columns_num
            image_height = block_size * rows_num
            inv_color = tuple(255 - val for val in base_color),

            checker_board = np.zeros((image_height, image_width, 3), np.uint8)

            color_row = 0
            color_column = 0

            for i in range(0, image_height, block_size):
                color_row = not color_row
                color_column = color_row

                for j in range(0, image_width, block_size):
                    checker_board[i:i + block_size, j:j +
                                                      block_size] = base_color if color_column else inv_color
                    color_column = not color_column
            return checker_board

        @staticmethod
        def fisheye_calibrate(imagelist):
            """
            find fisheye correction values from multiple images containing the checkerboard
            :Parameters: imagelist
            :Returns: image
            """

            # https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
            CHECKERBOARD = (10, 10)
            subpix_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
            calibration_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv.fisheye.CALIB_CHECK_COND + cv.fisheye.CALIB_FIX_SKEW
            objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
            objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
            _img_shape = None
            objpoints = []  # 3d point in real world space
            imgpoints = []  # 2d points in image plane.
            for img in imagelist:
                _img_shape = img.shape[:2]
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                # Find the chess board corners
                ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD,
                                                        cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
                # If found, add object points, image points (after refining them)
                if ret == True:
                    objpoints.append(objp)
                    cv.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
                    imgpoints.append(corners)
            N_OK = len(objpoints)
            K = np.zeros((3, 3))
            D = np.zeros((4, 1))
            rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
            tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
            rms, _, _, _, _ = \
                cv.fisheye.calibrate(
                    objpoints,
                    imgpoints,
                    gray.shape[::-1],
                    K,
                    D,
                    rvecs,
                    tvecs,
                    calibration_flags,
                    (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
                )
            print("Found " + str(N_OK) + " valid images for calibration")
            print("DIM=" + str(_img_shape[::-1]))
            print("K=np.array(" + str(K.tolist()) + ")")
            print("D=np.array(" + str(D.tolist()) + ")")

            dim = _img_shape[::-1]
            k = K.tolist()
            d = D.tolist()
            return dim, k, d

        @staticmethod
        def fisheye_correction(img, K, D, DIM):

            """
            correct for fisheye distortions
            :Parameters: image, camera matrix K, distortion matrix D, dimensions D
            :Returns: image
            """

            '''
            DIM=(1600, 1200)
            K = np.array(
                [[781.3524863867165, 0.0, 794.7118000552183], [0.0, 779.5071163774452, 561.3314451453386], [0.0, 0.0, 1.0]])
            D = np.array([[-0.042595202508066574], [0.031307765215775184], [-0.04104704724832258], [0.015343014605793324]])
            '''

            map1, map2 = cv.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv.CV_16SC2)

            undistorted_img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
            return undistorted_img

        @staticmethod
        def squared(img, leadingaxislargest=True):
            """Square a image based on smallest axis (width or height, cropped to a multiple of 2), for grayscale and color images

            :Parameters: image, leadingaxislargest=True
            :Returns: image
            """
            width = img.shape[1]
            height = img.shape[0]

            height = Image._multipleof2(height)
            width = Image._multipleof2(width)
            img = Image.crop(img, 0, 0, width, height)  # first make square

            if (leadingaxislargest == True):
                x = height if height > width else width
                y = height if height > width else width
                channels = len(img.shape)
                if (channels == 3):
                    square = np.zeros((x, y, 3), np.uint8)
                else:
                    square = np.zeros((x, y), np.uint8)
                square[int((y - height) / 2):int(y - (y - height) / 2),
                int((x - width) / 2):int(x - (x - width) / 2)] = img
            if (leadingaxislargest == False):
                x = height if height < width else width
                y = height if height < width else width
                square = Image.crop(img, 0, 0, x, y)
            return square

        @staticmethod
        def remove_whitespace(img):
            """Remove whitespace from an image. This can be used to remove a vignet.

            :Parameters: image
            :Returns: image
            """
            gray = Image.Convert.toGray(img)
            gray = Image.invert(gray)
            # gray = 255 * (gray < 128).astype(np.uint8)  # To invert the text to white
            coords = cv.findNonZero(gray)  # Find all non-zero points (text)
            x, y, w, h = cv.boundingRect(coords)  # Find minimum spanning bounding box
            rect = img[y:y + h, x:x + w]  # Crop the image - note we do this on the original image
            return rect

        @staticmethod
        def remove_blackspace(img):
            """Remove blackspace from an image. This can be used to remove a vignet.

            :Parameters: image
            :Returns: image
            """
            gray = Image.Convert.toGray(img)
            # gray = 255 * (gray < 128).astype(np.uint8)  # To invert the text to white
            coords = cv.findNonZero(gray)  # Find all non-zero points (text)
            x, y, w, h = cv.boundingRect(coords)  # Find minimum spanning bounding box
            rect = img[y:y + h, x:x + w]  # Crop the image - note we do this on the original image
            return rect

        @staticmethod
        def add_blackborder(img, bordersize=25):
            """Add black border to image

            :Parameters: image, bordersize
            :Returns: image
            """

            img_with_border = cv.copyMakeBorder(img, bordersize, bordersize, bordersize, bordersize, cv.BORDER_CONSTANT,
                                                value=[0, 0, 0])
            return img_with_border

        @staticmethod
        def add_border(img, bordersize=25, color=[255, 255, 255]):
            """Add border to image with color = [255,255,255]

            :Parameters: image, bordersize
            :Returns: image
            """

            img_with_border = cv.copyMakeBorder(img, bordersize, bordersize, bordersize, bordersize, cv.BORDER_CONSTANT,
                                                value=color)
            return img_with_border

        @staticmethod
        def add_blackmask(img, rect=[0, 0, 100, 100]):
            """Add mask to image

            :Parameters: image, rectangle = [x0,y0,x1,y1]
            :Returns: image
            """
            pt1 = (rect[0], rect[1])
            pt2 = (rect[2], rect[3])

            mask = np.zeros(img.shape, dtype=np.uint8)
            mask = cv.rectangle(mask, pt1, pt2, (255, 255, 255), -1)

            # Mask input image with binary mask
            result = cv.bitwise_and(img, mask)
            # Color background white
            # result[mask == 0] = 255  # Optional
            return result

        @staticmethod
        def imageregistration(img1, img2, verbose=False):
            """
            Register 2 images using opencv ORB (Oriented FAST and rotated BRIEF)
            Initiate ORB detector

            :Parameters: img1, img2
            :Returns: image
            """
            orb = cv.ORB_create(nfeatures=1000, scoreType=cv.ORB_HARRIS_SCORE)

            # Find the keypoints and descriptors with ORB
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)

            # BFMatcher with default params
            bf = cv.BFMatcher()

            # Apply ratio test
            pairMatches = bf.knnMatch(des1, des2, k=2)
            rawMatches = []
            for m, n in pairMatches:
                if m.distance < 0.7 * n.distance:
                    rawMatches.append(m)

            sortMatches = sorted(rawMatches, key=lambda x: x.distance)
            matches = sortMatches[0:128]

            img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:100], None, flags=2)
            if verbose == True:
                cv.imshow("Feature descriptor matching", img3)
                cv.waitKey(0)

            image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
            image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)

            for i in range(0, len(matches)):
                image_1_points[i] = kp1[matches[i].queryIdx].pt
                image_2_points[i] = kp2[matches[i].trainIdx].pt

            hom, mask = cv.findHomography(image_2_points, image_1_points, cv.RANSAC, ransacReprojThreshold=2.0)

            # Warp source image to destination based on homography
            im_out = cv.warpPerspective(img2, hom, (img2.shape[1], img2.shape[0]), flags=cv.INTER_LINEAR)
            return im_out

        @staticmethod
        def imageregistration_manual(img0, img1, pnts0, pnts1):
            """
            Image registration using Affine transform manually
            :Parameters: img0, img1, pnts0, pnts1
            :Returns: image
            """

            pnts = np.float32(pnts0)
            pnts2 = np.float32(pnts1)
            matrix = cv.getAffineTransform(pnts2, pnts)
            im_out = cv.warpAffine(img1, matrix, (img1.shape[1], img1.shape[0]))
            return im_out

        @staticmethod
        def video2thumbnailcontactsheet(fn_video, fn_contactsheet, rows=3, resize_factor=1):
            """Create a thumbnail contactsheet from a video

            :Parameters: filenamevideo,filenamecontactsheet,rows,resize_factor
            :Returns: image
            """
            cap = cv.VideoCapture(fn_video)
            fps = cap.get(cv.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
            frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            print(frame_count)
            stepsize = int(frame_count / (rows * rows))
            imagelist = []
            for i in range(0, (rows * rows)):
                cap.set(1, i * stepsize)
                ret, frame = cap.read()
                frame = Image.resize(frame, resize_factor)
                imagelist.append(frame)
            out = Image.Tools.patches2image(imagelist, cols=rows, overlap_px=0, whitebackground=False)
            Image.save(out, fn_contactsheet)

        @staticmethod
        def concat_two_images(imga, imgb):
            """
            Concat 2 images
            """

            imga = imga.astype(np.float32)
            imga = cv.normalize(imga, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC1)
            imgb = imgb.astype(np.float32)
            imgb = cv.normalize(imgb, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC1)

            # imga = imgb

            ha, wa = imga.shape[:2]
            hb, wb = imgb.shape[:2]
            max_height = np.max([ha, hb])
            total_width = wa + wb

            if len(imga.shape) < 3:
                new_img = np.zeros(shape=(max_height, total_width))
                new_img[:ha, :wa, ] = imga
                new_img[:hb, wa:wa + wb, ] = imgb
            else:
                new_img = np.zeros(shape=(max_height, total_width, 3))
                new_img[:ha, :wa, ] = imga
                new_img[:hb, wa:wa + wb, ] = imgb

            # dist2 = cv.convertScaleAbs(new_img)
            dist2 = cv.normalize(new_img, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC1)
            # dist2 = new_img

            return dist2

        @staticmethod
        def video_to_imagesondisk(file_in='video.avi', path_out='images'):
            """video to image

            :Parameters: video_filename
            :Returns: images
            """
            video_file = file_in
            output_folder = path_out
            vidcap = cv.VideoCapture(video_file)
            success, image = vidcap.read()
            count = 0
            success = True
            while success:
                fn = output_folder + "/" + "frame%d.png" % count
                cv.imwrite(fn, image)  # save frame as JPEG file
                success, image = vidcap.read()
                print('Read a new frame: ', success, fn)
                count += 1
            print("ready.")

        @staticmethod
        def imagesfromdisk_to_video(path_in, file_out='video.avi', framerate=15):
            """images from file to video

            :Parameters: path with list of frames
            :Returns: video
            """
            image_folder = path_in
            video_name = file_out

            output_folder = "output"

            fn = image_folder + "/" + output_folder + "/"
            print(fn)
            os.makedirs(os.path.dirname(fn), exist_ok=True)

            images = [img for img in os.listdir(image_folder) if (img.endswith(".tif") or img.endswith(".png"))]
            frame = cv.imread(os.path.join(image_folder, images[0]))
            height, width, layers = frame.shape

            video = cv.VideoWriter(fn + video_name, 0, framerate, (width, height))

            for image in images:
                video.write(cv.imread(os.path.join(image_folder, image)))

            cv.destroyAllWindows()
            video.release()

        @staticmethod
        def _get_dominant_color(img):
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

        @staticmethod
        def intensity_compensation_per_line(img):
            """intensity compensation per line

            :Parameters: image
            :Returns: image
            """

            col = Image.Tools._get_dominant_color(img)
            img2 = Image.Convert.toGray(img)
            img2 = cv.normalize(img2, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

            # calculate the mean intensity of each line
            mean_intensities = np.mean(img2, axis=1)
            # subtract the mean intensity from each line
            res = img2 - mean_intensities[:, np.newaxis]
            res = cv.normalize(res, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

            # Apply binary threshold to detect bright pixels
            _, thresh = cv.threshold(img2, 0, 128, cv.THRESH_BINARY)
            # Invert binary image
            # Multiply binary image with grayscale image to remove bright pixels
            res = cv.bitwise_and(res, thresh)
            res = Image.Process.Falsecolor.grayscale_to_color(res, col)
            return res

        @staticmethod
        def remove_features_at_boundaries(img, black_background=True):
            """Remove any of the features that are touching the boundaries

            :Parameters: image (grayscale 8bit)
            :Returns: image without features that touch the boundaries
            """

            if black_background == True:
                img = Image.Adjust.invert(img)
            height, width = img.shape[:2]
            mask = np.zeros((height + 2, width + 2), np.uint8)
            new_img = img.copy()
            for y in range(height):
                for x in range(width):
                    if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                        # Floodfill from boundary pixels with white color (255)
                        cv.floodFill(new_img, mask, (x, y), (255, 255, 255))
            # Invert the image so that the black pixels are the features to remove
            if black_background == True:
                result = Image.Adjust.invert(new_img)
            else:
                result = new_img
            return result

        @staticmethod
        def draw_color_bar():
            """Draw a gradient

            :Parameters: image, resize=0.15
            """
            gradient = np.arange(256, dtype=np.uint8).reshape(1, -1)
            colorbar = np.repeat(gradient, 50, axis=0)
            colorbar = Image.Convert.toRGB(colorbar)
            return colorbar

        @staticmethod
        def create_focus_stack_simple(images):
            """Create a simple focus stack

            :Parameters: image_list
            :returns: output_image
            """

            def do_lap(image, kernel_size=5, blur_size=5):
                blurred = cv.GaussianBlur(image, (blur_size, blur_size), 0)
                return cv.Laplacian(blurred, cv.CV_64F, ksize=kernel_size)

            laplacians = [do_lap(cv.cvtColor(image, cv.COLOR_BGR2GRAY)) for image in images]
            output = np.zeros_like(images[0])

            abs_laplacians = np.absolute(laplacians)
            max_indices = np.argmax(abs_laplacians, axis=0)

            for i in range(len(images)):
                output[max_indices == i] = images[i][max_indices == i]

            return output
