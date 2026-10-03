import os
import cv2
import sys
import pdb
import six
import glob
import time
import torch
import random
import pandas
import warnings
from dotenv import load_dotenv
import random
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
# import pyarrow as pa
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import video_augmentation

import torchvision.transforms as transforms
from dotenv import load_dotenv

if load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env")) == False:
    print("Error loading .env file for video2gloss module")

STORAGE_PATH = os.getenv("STORAGE_PATH")
#INPUT_FNAME = os.path.join(STORAGE_PATH, os.getenv("INPUT_FNAME"))
INPUT_FNAME ='./preprocess/captar-libras_10-12/test_info.npy'
sys.path.append("..")

class BaseFeeder(data.Dataset):
    def __init__(self, video_path, gloss_dict, drop_ratio=1.0, num_gloss=-1, transform_mode='test', save_frames = False, dataset= None):
        self.ng = num_gloss
        self.dict = gloss_dict
        self.data_type = 'video'
        self.video_path = video_path
        self.transform_mode = transform_mode.lower()
        self.save_frames = save_frames
        input_path = INPUT_FNAME
        self.inputs_list = np.load(input_path, allow_pickle=True).item()
        self.data_aug = self.transform()
        # print("")

    def __getitem__(self, idx):
        input_data = self.read_video()
        #input_data = [equalizeImg(frame) for frame in input_data]
        input_data = self.normalize(input_data, save_frames=self.save_frames)
        return input_data
        
    def read_video(self):
        print(f'Lendo arquivo: {self.video_path}')
        if self.video_path.endswith(('.mp4', '.avi', '.mkv')):            
            print('Lendo arquivo de vídeo...')
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file {self.video_path}")

            img_list = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                img_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
            cap.release()
            return img_list
        else:
            img_list = sorted(glob.glob(self.video_path + '*.png') + glob.glob(self.video_path + '*.jpg'))
        return [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list]

    def normalize(self, video, file_id=None, save_frames = True):
        video, _ = self.data_aug(video, file_id)
        
        if save_frames:
            img_dir = os.path.join('./teste_sys-dataloader', 'preprocessed/')
            
            os.makedirs(img_dir, exist_ok=True)
            for i, frame in enumerate(video.numpy().transpose((0, 2, 3, 1))):
                cv2.imwrite(os.path.join(img_dir, f'frame_{i:04d}.png'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) )  # Save image
                
        video = video.float()/255 
        
        means = torch.mean(video, axis=(0, 2, 3)) 
        stds = torch.std(video, axis=(0, 2, 3))         
        
        normalize = transforms.Compose([transforms.Normalize(mean=means.tolist(), std=stds.tolist())])
        
        video = normalize(video)
        return video

    def transform(self):
        # print("Apply testing transform.")
        #self.transform_mode = os.getenv("TRANSFORM_MODE")
        if self.transform_mode == "yolo":
            print("Applying YoloBBCrop")

            return video_augmentation.Compose([
                video_augmentation.YoloBBCrop(224),
                video_augmentation.ToTensor(),
            ])
        elif self.transform_mode == "yolo_hf":
            print("Applying YoloBBCrop With 15 fps")

            return video_augmentation.Compose([
                video_augmentation.HalfFps(),
                video_augmentation.YoloBBCrop(224),
                video_augmentation.ToTensor(),
            ])
        else:
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(224),
                video_augmentation.ToTensor(),
            ])
	
    @staticmethod
    def collate_fn(video):
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video])
            left_pad = 6
            right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
            print(f"{max_len = } {left_pad = } {right_pad = }")
            max_len = max_len + left_pad + right_pad
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
            
        return padded_video, video_length
            
    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


if __name__ == "__main__":
    
    INPUT_FNAME = '../preprocess/captar-libras_12-10/test_info.npy'
    gloss_dict = '../preprocess/captar-libras_12-10/gloss_dict.npy'
    
    
    video_paths = glob.glob('/srv/projects2/captarlibras_finep/sign-to-text/dataset/captar-libras_16-10/features/fullFrame-256x256px/*/*/')
    video_path = random.choice(video_paths)

    data_loader = BaseFeeder(video_path, gloss_dict, transform_mode='yolo', save_frames = True)

    print(video_path)
    for data in data_loader:
        print(data)
        break