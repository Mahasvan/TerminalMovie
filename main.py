from processing.video_processing import VideoProcessor
# from image_processing import ImageProcessor

import numpy as np

# file_path = input("Enter the path of the video: ")
file_path = "rec.mp4"
file_path = file_path.replace("'", "")

processor = VideoProcessor(file_path, 'frames')

fps, width, height, channels, image = processor.get_deets()

print(image[0][0])
print(image.shape)
image = image.mean(axis=2).astype(int)

# import matplotlib.pyplot as plt
# plt.imshow(image, cmap='gray')
# plt.show()

sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
sobel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

def convolve(image, kernel):
    kernel = np.array(kernel)
    output = np.zeros_like(image)
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            output[i, j] = (image[i-1:i+2, j-1:j+2] * kernel).sum()
    return output
