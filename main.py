from video_processing import VideoProcessor
from image_processing import ImageProcessor

import os, json

file_path = input("Enter the path of the video: ")
file_path = file_path.replace("'", "")
save_path = input("Enter the path to save the frames: ")
save_path = save_path.replace("'", "")

if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists(os.path.join(save_path, "frame0.jpg")):
    vp = VideoProcessor(file_path, save_path)
    print("Splitting into frames")
    vp.split_to_frames()
    print("Split! Frames: ", vp.frame_count)

print("Processing frames")
frames = os.listdir(save_path)
frames.sort(key=lambda x: int(x.replace("frame", "").replace(".jpg", "")))

processed_frames_path = os.path.join(save_path, "processed")
if not os.path.exists(processed_frames_path):
    os.makedirs(processed_frames_path, exist_ok=True)

matrix_path = os.path.join(save_path, "matrices")
if not os.path.exists(matrix_path):
    os.makedirs(matrix_path, exist_ok=True)

for frame in frames:
    print("Processing", frame)
    source_path = os.path.join(save_path, frame)
    dest_path = os.path.join(processed_frames_path, frame)
    matrix_dest_path = os.path.join(matrix_path, frame.replace(".jpg", ".json"))
    ip = ImageProcessor(source_path, dest_path)
    ip.remove_background()
    ip.downsample()
    matrix = ip.convert_to_xterm()
    with open(matrix_dest_path, 'w') as f:
        json.dump(matrix, f)
