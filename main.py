from processing.video_processing import VideoProcessor
# from image_processing import ImageProcessor

import numpy as np

# file_path = input("Enter the path of the video: ")
file_path = "try.mp4"

processor = VideoProcessor(file_path, 'frames', output_path="outs")
# processor.split_to_frames()
# processor.get_edges(downsample=2)
# processor.get_full_frame(downsample=4)
processor.print(speed=1, combined=True)
