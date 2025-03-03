import PIL.Image
import cv2
import numpy
from PIL import Image, ImageFilter

import numpy as np

from collections import Counter

class ImageLoader:
    def __init__(self, path: str):
        self.path = path
        self.pil_image_color = PIL.Image.open(path)
        self.numpy_color = numpy.array(self.pil_image_color)

        self.pil_image = self.pil_image_color.convert('L')
        self.numpy = np.array(self.pil_image)

def get_char(angle):
    if -181 <= angle <= -157.5:
        return "－"
    if -157.5 <= angle <= -112.5:
        return "／"
    if -112.5 <= angle <= -67.5:
        return "｜"
    if -67.5 <= angle <= -22.5:
        return "＼"
    if -22.5 <= angle <= 22.5:
        return "－"
    if 22.5 <= angle <= 67.5:
        return "／"
    if 67.5 <= angle <= 112.5:
        return "｜"
    if 112.5 <= angle <= 157.5:
        return "＼"
    if 157.5 <= angle <= 181.0:
        return "－"

    else:
        return "　"

class Circle:
    def __init__(self, image_width, radius):
        self.image_width = image_width
        self.radius = radius
        self.centre = (image_width//2, image_width//2)

        self.image = numpy.zeros((self.image_width, self.image_width))

        for i in range(image_width):
            for j in range(image_width):
                if self._check_bound(i, j):
                    self.image[i, j] = 255

    def _check_bound(self, i, j):
        centre_x, centre_y = self.centre
        return (i-centre_x)**2 + (j-centre_y)**2 <= self.radius**2

    def load_numpy(self):
        return self.image






class ImageProcessor:
    def __init__(self, image_path: str):
        self.edge_kernel = ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8,
                                                       -1, -1, -1, -1), 1, 0)
        self.sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.sobel_y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.image_path = image_path
        self.image_loader = ImageLoader(image_path)

        self.contour_threshold = 25
        self.sobel_kernel_size = 3

    def downsample(self, chunk_size=4):
        image = self.image_loader.numpy.copy()
        new = image[::chunk_size, ::chunk_size]
        return PIL.Image.fromarray(new)

    def find_edges(self):
        # img_1 = np.array(image.filter(ImageFilter.GaussianBlur(radius=2)))
        # img_2 = np.array(image.filter(ImageFilter.GaussianBlur(radius=1)))
        image = self.image_loader.pil_image

        return np.array(image.filter(ImageFilter.FIND_EDGES))
        # return np.array(image.filter(self.edge_kernel))
        # return img_2 - img_1


    def sobel(self):
        image = self.image_loader.numpy

        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel_size)
        sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=self.sobel_kernel_size)

        return sobelx, sobely


    def edge_highlighting(self, downsample: int = 4):

        edges = self.find_edges()
        result_x, result_y = self.sobel()
        # print("Calculated Sobel")
        angles = np.arctan2(result_x, result_y)

        edges = edges[::downsample, ::downsample]
        angles = angles[::downsample, ::downsample]

        scaled_angles = angles * 180 / np.pi
        chars = np.zeros_like(scaled_angles).astype(str)
        for i in range(scaled_angles.shape[0]):
            for j in range(scaled_angles.shape[1]):
                if edges[i, j] >= self.contour_threshold:
                    chars[i, j] = get_char(scaled_angles[i, j])
                else:
                    chars[i, j] = "　"
        return chars

    def intensity_char_mapping(self, downsample: int = 4):
        chars = "　．，：～０＆％＃＠"
        limits = [255/len(chars) * x for x in range(1,11)]
        # print(limits)
        # ten values, steps of 25.5

        image = self.image_loader.numpy[::downsample, ::downsample]

        res = numpy.zeros_like(image).astype(str)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k, c in enumerate(chars):
                    if image[i, j] <= limits[k]:
                        res[i, j] = c
                        break
        return res

    def asciify(self, downsample: int = 4):
        contours = self.edge_highlighting(downsample)
        intensities = self.intensity_char_mapping(downsample)
        for i in range(intensities.shape[0]):
            for j in range(intensities.shape[1]):
                if contours[i, j] != "　":
                    intensities[i, j] = contours[i,j]

        return intensities

    def pooling(self, downsample: int = 4):
        image = self.image_loader.numpy_color
        rows, cols = image.shape[0]//downsample, image.shape[1]//downsample
        res = np.zeros((rows, cols, image.shape[2]), dtype=self.image_loader.numpy.dtype)

        for i in range(rows):
            for j in range(cols):
                pixels = image[i*downsample:i*downsample+downsample, j*downsample:j*downsample+downsample]
                unique, counts = np.unique(pixels, return_counts=True, axis=1)
                least_common = unique[np.argmax(counts)][0]
                res[i, j] = least_common
        return res
