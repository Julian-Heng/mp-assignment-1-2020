#!/usr/bin/env python3

from functools import wraps
from matplotlib import pyplot as plt
from pathlib import Path

import cv2
import numpy as np


class Image:

    """
    List of valid operations:
        brightness
        center
        connected_components
        crop
        gaussian
        harris
        histogram
        hog
        kmeans
        rgb_mute
        normalize
        reset
        resize
        rotate
        sift_descriptors
        sift_keypoints
        square
        threshold
        write
    """

    COLORS = ("blue", "green", "red")
    COLORS_VAL = {
        "blue": [255, 0, 0],
        "green": [0, 255, 0],
        "red": [0, 0, 255],
    }

    COLOR_BGR = 0
    COLOR_GRAY = 1

    def __init__(self, filename):
        self._original_filename = filename
        self._filename = Path(self._original_filename).name

        self._original_image = cv2.imread(self._original_filename)
        self._image = self._original_image.copy()
        self._color_space = Image.COLOR_BGR
        self._print_write_callback = None

    @property
    def shape(self):
        """ Returns the shape of the current image """
        return self._image.shape[:2]

    @property
    def width(self):
        """ Returns the width of the image """
        return self.shape[1]

    @property
    def height(self):
        """ Returns the height of the image """
        return self.shape[0]

    @property
    def filename(self):
        """ Returns the filename of the image """
        return self._filename

    @property
    def print_write_callback(self):
        """ Returns the print callback function of this class """
        return self._print_write_callback

    @print_write_callback.setter
    def print_write_callback(self, value):
        """ Sets the print callback function for this class """
        self._print_write_callback = value

    def _handle_fname(method):
        """ Method decorator that handles the fname and append parameters """

        @wraps(method)
        def inner(self, *args, **kwargs):
            # Set up the fname variable using the append
            name = method.__name__
            fname = kwargs.pop("fname", self._original_filename)
            append = kwargs.pop("append", None)
            append = name if append is None else "{}_{}".format(append, name)

            # Place fname back into arguments
            fname = Image._append_filename(fname, append)
            kwargs["fname"] = fname

            # Call method
            return method(self, *args, **kwargs)

        return inner

    def brightness(self, value=1.0):
        """
        Increases the brightness of the image

        Wrapper around the opencv convertScaleAbs function
        From: https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
        Accessed: 29/9/2020
        """
        self._image = cv2.convertScaleAbs(self._image, alpha=1.0, beta=value)
        return self

    def center(self, size=0):
        """ Crops the border of an image denoted by size """
        x = size
        y = size
        w, h = self._image.shape[:2]
        w -= size * 2
        h -= size * 2
        return self.crop(x, y, w, h)

    @_handle_fname
    def connected_components(self, count=False, fname=None):
        """
        Performs a connected component label operation on the image

        Wrapper around the opencv connectedComponentsWithStats function
        From: https://docs.opencv.org/3.4.2/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f
        Accessed: 13/9/2020
        """

        def _get_area(stats, label):
            """ Prints the area of a component in a pretty manner """
            fmt = "component {}: {} pixels"
            return fmt.format(label, stats[label][cv2.CC_STAT_AREA])

        # Set up output directory
        basedir = Path(fname)
        basedir = Path(basedir.parent).joinpath(basedir.stem)
        basedir.mkdir(parents=True, exist_ok=True)

        # Get components
        count, labels, stats, _ = cv2.connectedComponentsWithStats(self._image)

        # Process each component
        for label in range(0, count):
            # Get the component with this label
            mask = np.array(labels, dtype=np.uint8)
            mask[labels == label] = 255
            mask[labels != label] = 0

            # Crop out the component using a bounding rectangle
            x, y, w, h = cv2.boundingRect(mask)
            mask = mask[y : y + h, x : x + w]

            # Write to file
            label_append = "original" if label == 0 else label
            fname = self._filename
            fname = Image._append_filename(fname, "ccl_{}".format(label_append))
            fname = Path(basedir).joinpath(fname)

            Image._write(mask, str(fname), self._print_write_callback)

        # Write the statistics of the components
        if count:
            stat_fname = basedir.joinpath("stats.txt")
            with open(str(stat_fname), "w") as f:
                f.write("No. of components: {}\n".format(count - 1))
                f.write("\n".join([_get_area(stats, i) for i in range(1, count)]))

        return self

    def crop(self, x, y, w, h):
        """ Performs a crop operation on the image """
        if any(i < 0 for i in (x, y, w, h)):
            raise TypeError("coordinates or dimensions cannot be negative")
        if any(i == 0 for i in (w, h)):
            raise TypeError("dimensions cannot be zero")

        self._image = self._image[x : x + w, y : y + h]
        return self

    def rgb_mute(self, blue=False, green=False, red=False):
        """ Removes a color channel from the image """
        if blue:
            self._image[:, :, 0] = 0
        if green:
            self._image[:, :, 1] = 0
        if red:
            self._image[:, :, 2] = 0
        return self

    def gaussian(
        self, ksize=(5, 5), sigma_x=0, sigma_y=0, border_type=cv2.BORDER_DEFAULT
    ):
        """
        Performs a gaussian blur filter on the image

        Wrapper around the opencv GaussianBlur function
        From: https://docs.opencv.org/3.4.2/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
        Accessed: 19/9/2020
        """
        self._image = cv2.GaussianBlur(
            self._image, ksize, sigma_x, sigma_y, border_type
        )
        return self

    def harris(self, block_size=2, kernel_size=3, k=0.04, threshold=0.05, color="red"):
        """
        Performs a harris corner algorithm on the image

        Wrapper around the opencv cornerHarris function
        From: https://docs.opencv.org/3.4.2/dd/d1a/group__imgproc__feature.html#gac1fc3598018010880e370e2f709b4345
        Accessed: 13/9/2020
        """
        # Input validation
        color = color.lower()
        if color not in Image.COLORS_VAL.keys():
            raise TypeError("color parameter is invalid")
        color = Image.COLORS_VAL[color]

        # Prepare image
        img = self._image.copy()

        gray = Image._grayscale(img, self._color_space)
        gray = np.float32(gray)

        # Get harris detector output
        dst = cv2.cornerHarris(gray, block_size, kernel_size, k)

        # Highlight colors
        img[dst > threshold * dst.max()] = color

        self._image = img
        return self

    @_handle_fname
    def histogram(self, bins=10, colors=None, fname=None):
        """
        Graphs the color information of the original image and this image to a
        histogram

        From: https://docs.opencv.org/master/d1/db7/tutorial_py_histogram_begins.html
        Accessed: 12/9/2020
        """
        # Input validation
        if colors is None:
            colors = Image.COLORS
        elif not set(Image.COLORS).issuperset(set(colors)):
            raise TypeError("colors parameter is invalid")

        img = self._image.copy()

        # Setup figures and subplots
        num_cols = len(colors)
        fig, axes = plt.subplots(1, num_cols, figsize=(3 * num_cols, 3))
        fig.tight_layout(pad=4.0)
        fig.suptitle(fname)

        # For each color
        for column, color in enumerate(colors):
            # Get color information
            hist = img[:, :, column].reshape(-1)

            # Plot the color
            axes[column].hist(
                hist, bins, edgecolor="black", color=color[0]
            )

            # Set axes information
            axes[column].set_title(color)
            axes[column].set_xlabel("Intensity")
            axes[column].set_ylabel("No. of pixels")
            axes[column].set_xlim([0, 256])

        # Save figure
        Image._plt_savefig(fname)
        plt.clf()
        plt.close()

        return self

    @_handle_fname
    def hog(self, fname=None):
        """
        Performs a Histogram of Oriented Gradients operation on the image

        From: https://stackoverflow.com/questions/6090399/get-hog-image-features-from-opencv-python
        Accessed: 15/9/2020
        """
        # Prepare image
        orig = self._original_image.copy()
        img = self._image.copy()

        orig_gray = Image._grayscale(orig, Image.COLOR_BGR)
        gray = Image._grayscale(img, self._color_space)

        # Get hog
        hog = cv2.HOGDescriptor()

        win_stride = (8, 8)
        padding = (8, 8)
        locations = ((10, 20),)

        # Compute descriptors for the original image and current image
        orig_des = hog.compute(orig_gray, win_stride, padding, locations)
        des = hog.compute(gray, win_stride, padding, locations)

        # Normalise the histograms
        orig_des = cv2.normalize(orig_des, None)
        des = cv2.normalize(des, None)

        # Compare the 2 histograms
        diff = cv2.norm(orig_des, des)

        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        fig.tight_layout(pad=3.0)

        fig.suptitle("Difference: {:.02f}".format(diff))

        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title(fname)
        axes[1].hist(des, edgecolor="black")
        axes[1].set_title("HOG Histogram")

        Image._plt_savefig(fname)
        plt.clf()
        plt.close()

        return self

    @_handle_fname
    def kmeans(self, k=1, fname=None):
        """
        Performs KMeans clustering on the image for image segmentation

        Wrapper around the opencv kmeans function
        From: https://docs.opencv.org/master/d1/d5c/tutorial_py_kmeans_opencv.html
        Accessed: 27/9/2020

        From: https://docs.opencv.org/3.4.2/d5/d38/group__core__cluster.html#ga9a34dc06c6ec9460e90860f15bcd2f88
        Accessed: 27/9/2020

        From: https://www.thepythoncode.com/article/kmeans-for-image-segmentation-opencv-python
        Accessed: 27/9/2020
        """
        # Prepare image
        img = self._image.copy()
        if self._color_space == Image.COLOR_BGR:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self._color_space == Image.COLOR_HSV:
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        elif self._color_space == Image.COLOR_GRAY:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = self._image

        # Convert image into a single dimension array of floats
        pixel_values = img.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        # Define the criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        # Perform kmeans
        ret, labels, centers = cv2.kmeans(
            pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        # Convert back to integers
        centers = np.uint8(centers)

        # Flatten the labels array
        labels = labels.reshape(-1)

        # Reshape the segmented image
        segmented_img = centers[labels]
        segmented_img = segmented_img.reshape(img.shape)

        # Write the segmented image to file
        Image._write(segmented_img, fname, self._print_write_callback)

        # For each cluster
        for cluster in range(0, k):
            # Set filename
            this_fname = Image._append_filename(fname, "cluster_{}".format(cluster))

            # Use the original image as the base image
            mask = self._original_image.copy()

            # Blacken everything but the cluster
            mask = mask.reshape((-1, 3))
            mask[labels == cluster] = [0, 0, 0]
            mask = mask.reshape(img.shape)

            # Write masked image to file
            Image._write(mask, this_fname, self._print_write_callback)

        return self

    def normalize(self, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=-1):
        """
        Normalises the colors of the image

        Wrapper around the opencv normalize function
        From: https://docs.opencv.org/3.4.2/dc/d84/group__core__basic.html#ga1b6a396a456c8b6c6e4afd8591560d80
        Accessed: 19/9/2020
        """
        norm = np.zeros(self._image.shape[:2])
        norm = cv2.normalize(self._image, norm, alpha, beta, norm_type, dtype)
        self._image = norm
        return self

    def reset(self):
        """ Resets the image back to the original image """
        self._image = self._original_image.copy()
        self._color_space = Image.COLOR_BGR
        return self

    def resize(self, factor=(1.0, 1.0)):
        """
        Resizes the image to a new size

        Wrapper around the opencv resize function
        From: https://docs.opencv.org/3.4.2/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
        Accessed: 13/9/2020
        """
        if isinstance(factor, float):
            factor = (factor, factor)
        factor = factor[:2]
        if any(i < 0.0 for i in factor):
            raise TypeError("size factor cannot be negative")

        img = self._image.copy()

        resize = (self.width, self.height)
        resize = tuple(int(i * j) for i, j in zip(resize, factor))
        img = cv2.resize(img, resize)

        self._image = img
        return self

    def rotate(self, angle=0.0):
        """
        Rotates the image

        From: https://stackoverflow.com/a/37347070
        Accessed: 13/9/2020
        """
        img = self._image.copy()

        # Get image center
        center = (self.width / 2, self.height / 2)
        rot = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Get new image size after rotation
        abs_cos = abs(rot[0, 0])
        abs_sin = abs(rot[0, 1])

        bound_w = int(self.height * abs_sin + self.width * abs_cos)
        bound_h = int(self.height * abs_cos + self.width * abs_sin)
        bounds = (bound_w, bound_h)

        # Fix rotation matrix
        rot[0, 2] += (bound_w / 2) - center[0]
        rot[1, 2] += (bound_h / 2) - center[1]

        # Do transformation
        img = cv2.warpAffine(img, rot, bounds)
        self._image = img
        return self

    def sift_descriptors(self):
        """
        Compares the sift descriptors between the original image and the
        current image

        From: https://blog.francium.tech/feature-detection-and-matching-with-opencv-5fd2394a590
        Accessed: 27/9/2020
        """
        orig = self._original_image.copy()
        img = self._image.copy()

        orig_kp, orig_des = Image._sift(orig, Image.COLOR_BGR)
        img_kp, img_des = Image._sift(img, self._color_space)

        if all(x is not None for x in (orig_des, img_des)):
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = bf.match(orig_des, img_des)
            matches = sorted(matches, key=lambda x: x.distance)

            # Draw the best match
            self._image = cv2.drawMatches(
                orig, orig_kp, img, img_kp, matches[0:1], img,
                matchColor=Image.COLORS_VAL["green"], flags=2
            )

        return self

    def sift_keypoints(self):
        """
        Draws the sift keypoints to an image

        From: https://docs.opencv.org/3.4.2/da/df5/tutorial_py_sift_intro.html
        Accessed: 13/9/2020
        """
        img = self._image.copy()
        kp, _ = Image._sift(img, self._color_space)
        out = None
        out = cv2.drawKeypoints(img, kp, out)
        self._image = out
        return self

    def square(self):
        """ Crops the image into a square """
        w, h = self._image.shape[:2]
        if w < h:
            delta = (h - w) // 2
            size = h - (delta * 2)
            self.crop(0, delta, w, size)
        else:
            delta = (w - h) // 2
            size = w - (delta * 2)
            self.crop(delta, 0, size, h)
        return self

    def threshold(self, threshold=0.0, maxval=255.0, threshold_type=cv2.THRESH_BINARY):
        """
        Applies a threshold on the image

        Wrapper around the opencv threshold function
        From: https://docs.opencv.org/3.4.2/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
        Accessed: 13/9/2020
        """
        gray = Image._grayscale(self._image, self._color_space)
        _, bin_img = cv2.threshold(gray, threshold, maxval, threshold_type)

        self._image = bin_img
        return self

    def write(self, img=None, fname=None, append=None):
        """ Writes image to file """
        if img is None:
            img = self._image
        if fname is None:
            fname = self._filename
        if append is not None:
            fname = Image._append_filename(fname, append)
        Image._write(img, fname, self._print_write_callback)
        return self

    @staticmethod
    def _append_filename(fname, name):
        """ Static method that creates a new filename """
        p = Path(fname)
        return "{}_{}{}".format(p.stem, name, p.suffix)

    @staticmethod
    def _grayscale(img, color_space):
        """ Static method that converts the image to grayscale """
        if color_space == Image.COLOR_BGR:
            ret = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            return img

        return ret

    @staticmethod
    def _sift(img, color_space):
        """ Static method that performs SIFT """
        gray = Image._grayscale(img, color_space)
        sift = cv2.xfeatures2d.SIFT_create()
        return sift.detectAndCompute(img, None)

    @staticmethod
    def _write(img, fname, print_callback=None):
        """ Static method that writes an image to file """
        if print_callback is not None:
            print_callback(fname)
        cv2.imwrite(fname, img)

    @staticmethod
    def _plt_savefig(fname, print_callback=None):
        """ Static method that saves a figure to file """
        if print_callback is not None:
            print_callback(fname)
        plt.savefig(fname)
