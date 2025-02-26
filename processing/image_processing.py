import PIL.Image
import numpy
from PIL import Image, ImageFilter

import numpy as np

class ImageLoader:
    def __init__(self, path: str):
        self.path = path
        self.numpy = np.array(PIL.Image.open(path))

    def load_numpy(self):
        return self.numpy

def get_char(angle):
    if -180 <= angle <= -157.5:
        return "－"
    if -157.5 <= angle <= -112.5:
        return "＼"
    if -112.5 <= angle <= -67.5:
        return "｜"
    if -67.5 <= angle <= -45:
        return "／"
    if -45 <= angle <= -22.5:
        return "／"
    if -22.5 <= angle <= 22.5:
        return "－"
    if 22.5 <= angle <= 45:
        return "＼"
    if 45 <= angle <= 90:
        return "｜"
    if 90 <= angle <= 135:
        return "／"
    if 135 <= angle <= 180:
        return "－"
    else: return "　"

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

    def downsample(self, chunk_size=4):
        image = np.array(Image.open(self.image_path))
        new = image[::chunk_size, ::chunk_size]
        return PIL.Image.fromarray(new)

    def find_edges(self):
        # img_1 = np.array(image.filter(ImageFilter.GaussianBlur(radius=2)))
        # img_2 = np.array(image.filter(ImageFilter.GaussianBlur(radius=1)))
        image = Image.open(self.image_path)
        image = image.convert('L')
        return np.array(image.filter(self.edge_kernel))
        # return img_2 - img_1

    def sobel(self):
        image = Image.open(self.image_path)
        image = image.convert('L')
        image = np.array(image).astype("float")
        x_output = np.zeros_like(image)
        y_output = np.zeros_like(image)
        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
                x_output[i, j] = (image[i-1:i+2, j-1:j+2] * self.sobel_x_kernel).sum()
                y_output[i, j] = (image[i-1:i+2, j-1:j+2] * self.sobel_y_kernel).sum()
        return x_output, y_output

    def expand_edge(self, matrix):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] > 50:
                    matrix[i, j] = 1
                else:
                    matrix[i, j] = 0

        res = np.zeros_like(matrix)
        neighbours = [(0,1), (0,-1), (1,0), (1,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
        for i in range(1, matrix.shape[0]-1):
            for j in range(1, matrix.shape[1]-1):
                if matrix[i, j] == 1:
                    res[i, j] = 0
                # check if neighbours are 1
                for dx, dy in neighbours:
                    if matrix[i+dx, j+dy] == 1:
                        res[i, j] = 255
                        break
        return res


    def edge_highlighting(self, downsample: int = 4):

        edges = self.find_edges()
        result_x, result_y = self.sobel()
        # print("Calculated Sobel")
        angles = np.arctan2(result_x, result_y)

        edges = edges[::downsample, ::downsample]
        # edges = self.expand_edge(edges)
        angles = angles[::downsample, ::downsample]

        scaled_angles = angles * 180 / np.pi
        chars = np.zeros_like(scaled_angles).astype(str)
        for i in range(scaled_angles.shape[0]):
            for j in range(scaled_angles.shape[1]):
                if edges[i, j] >= 50:
                    chars[i, j] = get_char(scaled_angles[i, j])
                else:
                    chars[i, j] = "　"
        return chars

    def intensity_char_mapping(self, downsample: int = 4):
        chars = "　．，：～０＆％＠＃"
        limits = [255/len(chars) * x for x in range(1,11)]
        # print(limits)
        # ten values, steps of 25.5

        image = Image.open(self.image_path)
        image = image.convert('L')
        image = np.array(image).astype("float")
        image = image[::downsample, ::downsample]

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