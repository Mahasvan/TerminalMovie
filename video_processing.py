import cv2
import os

class VideoProcessor:
    def __init__(self, file_path, frame_path) -> None:
        self.path = file_path
        self.frame_path = frame_path
        self.frame_count = 0

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
            self.frame_count += 1
            count += 1