import json
import math

import cv2
import os
from processing.image_processing import ImageProcessor

import time
from processing.shell import RED, END_FORMATTING

class VideoProcessor:
    def __init__(self, file_path, frame_path, output_path) -> None:
        self.path = file_path
        self.frame_path = frame_path
        self.output_path = output_path

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)
        if not os.path.exists(self.frame_path):
            os.makedirs(self.frame_path, exist_ok=True)

        self.frame_count = 0

        self.fps, self.height,self.width, self.frame_count, self.image = self.get_deets()


    def split_to_frames(self):
        # Split video to frames
        vidcap = cv2.VideoCapture(self.path)
        success,image = vidcap.read()
        count = 0
        save_path = os.path.join(self.frame_path, 'frame')
        while success:
            cv2.imwrite(save_path + f"{count}.jpg", image)     # save frame as JPEG file      
            print("Saved to:", save_path + f"{count}.jpg")
            success,image = vidcap.read()
            count += 1

    def get_deets(self):
        vidcap = cv2.VideoCapture(self.path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        success, image = vidcap.read()
        height, width, layers = image.shape

        return fps, height, width, frames, image

    def get_edges(self, downsample=4):
        frames = os.listdir(self.frame_path)
        n_digits = int(math.log(self.frame_count, 10)) + 1

        print(len(frames))
        for frame in frames:
            number = frame.lstrip("frame").rstrip(".jpg")
            processor = ImageProcessor(os.path.join(self.frame_path, frame))
            edges = processor.edge_highlighting(downsample=downsample)
            with open(os.path.join(self.output_path, f"edges{int(number):0{n_digits}d}.txt"), 'w') as f:
                for i in range(edges.shape[0]):
                    f.write("".join(list(edges[i, :])))
                    f.write("\n")

    def get_full_frame(self, downsample=4):
        frames = os.listdir(self.frame_path)
        n_digits = int(math.log(self.frame_count, 10)) + 1

        print(len(frames))
        for frame in frames:
            number = frame.lstrip("frame").rstrip(".jpg")
            processor = ImageProcessor(os.path.join(self.frame_path, frame))
            edges = processor.asciify(downsample=downsample)
            final_name = f"combined{int(number):0{n_digits}d}.txt"
            with open(os.path.join(self.output_path, final_name), 'w') as f:
                for i in range(edges.shape[0]):
                    f.write("".join(list(edges[i, :])))
                    f.write("\n")
            print("Saved to:", os.path.join(self.output_path, final_name))

    def print(self, speed=1, combined=False, edges=False):
        if combined:
            files = [file for file in os.listdir(self.output_path) if file.startswith("combined")]
        elif edges:
            files = [file for file in os.listdir(self.output_path) if file.startswith("edges")]
        else:
            files = [file for file in os.listdir(self.output_path) if file.startswith("intensity")]

        files.sort()
        fps = self.fps * speed
        data = []
        for file in files:
            with open(os.path.join(self.output_path, file), "r") as f:
                data.append(f.read())

        history = []
        for frame in data:
            start = time.time()
            os.system("cls") if os.name == "nt" else os.system("clear")
            print(RED, end="")
            print(frame)
            end = time.time()
            diff = end - start
            history.append(diff)
            time.sleep(max(1/fps-diff, 0))
        with open(os.path.join(self.output_path, 'history.txt'), 'w') as f:
            json.dump(history, f, indent=2)