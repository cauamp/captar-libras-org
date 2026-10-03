import glob
import numpy as np
import cv2
from ..utils import video_augmentation
from tqdm import tqdm

crop = video_augmentation.YoloBBCrop(224)

paths = [
    "../datasets_srv/testes_cam_caua/*.mp4",
    "../datasets_srv/testes_cam_c922_fix/*.mkv",
    "../datasets_srv/testes_cam_zed/*.avi",
    "../datasets_srv/testes_cam_brio/*.mp4",
    "../datasets_srv/testes_cam_c922_default/*.mkv",
    "../datasets_srv/testes_cam_notebook/*.mp4",
]


def process_video(video_file):
    # print(f"Processing video {video_file}\n")
    imgs = []
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_file}")
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        imgs.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # print("\rFrame %d processed" % (i), end="")
        i += 1
    cap.release()
    imgs = np.array(imgs)
    imgs = crop(imgs)
    means = np.mean(imgs / 255, axis=(0, 1, 2))
    stds = np.std(imgs / 255, axis=(0, 1, 2))
    return means, stds


def process_videos_in_path(path):
    videos = glob.glob(path)
    data = {"means": [], "std": []}

    for video in tqdm(videos, desc=f"Processing videos in {path}"):
        means, stds = process_video(video)
        data["means"].append(means)
        data["std"].append(stds)

    overall_mean = np.mean(np.array(data["means"]), axis=0)
    overall_std = np.mean(np.array(data["std"]), axis=0)

    print(f"Overall mean for path {path}: {overall_mean}")
    print(f"Overall std for path {path}: {overall_std}")

    with open("./02-04_mean_std.txt", "a") as f:
        f.write(f"{path}\n")
        f.write(f"Overall mean: {overall_mean}\n")
        f.write(f"Overall std: {overall_std}\n")
        f.write("\n")


for path in paths:
    process_videos_in_path(path)
