import os
import cv2
import sys

sys.path.append(os.path.abspath("./"))
sys.path.append("..")
import six
import glob
import time
import torch
import warnings
from dotenv import load_dotenv
import concurrent.futures


import numpy as np
from PIL import Image
import torch.utils.data as data
from utils import video_augmentation
from utils.kornia_cam import create_camera_simulation_pipeline

import torchvision.transforms as transforms

warnings.simplefilter(action="ignore", category=FutureWarning)


# Load environment variables from .env file
load_dotenv("./dataset/config.env")

# Retrieve environment variables
DROP_RATIO = float(os.getenv("DROP_RATIO"))
NUM_GLOSS = int(os.getenv("NUM_GLOSS"))
MODE = os.getenv("MODE")
TRANSFORM_MODE = os.getenv("TRANSFORM_MODE").lower() in ("true", "1", "t")
DATATYPE = os.getenv("DATATYPE")
INPUTS_LIST_PATH = os.getenv("INPUTS_LIST_PATH")
FEATURES_PATH = os.getenv("FEATURES_PATH")
IMG_SIZE = int(os.getenv("IMG_SIZE"))
AUGMENT_RANDOM_CROP = int(os.getenv("AUGMENT_RANDOM_CROP"))
AUGMENT_HORIZONTAL_FLIP = float(os.getenv("AUGMENT_HORIZONTAL_FLIP"))
AUGMENT_TEMPORAL_RESCALE = float(os.getenv("AUGMENT_TEMPORAL_RESCALE"))
AUGMENT_CENTER_CROP = int(os.getenv("AUGMENT_CENTER_CROP"))
GLOSS_DICT_PATH = os.getenv("GLOSS_DICT_PATH")
PREFIX = os.getenv("PREFIX")
NPZ_FOLDER = os.getenv("INTEGER_NPZ_FOLDER")
# TORCH_MEAN = [float(x) for x in os.getenv('TORCH_MEAN').split(',')]
# TORCH_STD = [float(x) for x in os.getenv('TORCH_STD').split(',')]


class BaseFeeder(data.Dataset):
    def __init__(
        self,
        prefix,
        gloss_dict,
        drop_ratio=DROP_RATIO,
        num_gloss=NUM_GLOSS,
        mode=MODE,
        transform_mode=TRANSFORM_MODE,
        datatype=DATATYPE,
        kornia_kwargs={},
    ):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.dict = gloss_dict
        self.data_type = datatype
        self.feat_prefix = f"{prefix}/features/fullFrame-{IMG_SIZE}x{IMG_SIZE}px/{mode}"
        self.npz_folder = f"{NPZ_FOLDER}{mode}"
        os.makedirs(self.npz_folder, exist_ok=True)

        self.kornia_kwargs = kornia_kwargs
        self.transform_mode = transform_mode if transform_mode else "test"
        self.inputs_list = np.load(
            INPUTS_LIST_PATH.format(mode=mode), allow_pickle=True
        ).item()
        print(mode, len(self))
        self.data_aug = self.transform()
        self.to_tensor = self.to_tensor()
        print("")

    def __getitem__(self, idx):
        if self.data_type == "video":
            file_path = os.path.join(
                self.npz_folder, f"{self.inputs_list[idx]['folder'].split('/')[-2]}.npz"
            )
            if os.path.exists(file_path):
                try:
                    data = np.load(file_path)
                    input_data = data["video"]
                    label = data["label"].tolist()
                except Exception as e:
                    print(f"Erro ao carregar o arquivo {file_path}: {e}")
                    input_data, label, _ = self.read_video(idx)
                    input_data = np.array(input_data)
                    self.save_npz(
                        input_data,
                        label,
                        self.inputs_list[idx]["folder"].split("/")[-2],
                    )
            else:
                input_data, label, _ = self.read_video(idx)
                input_data = np.array(input_data)
                self.save_npz(
                    input_data, label, self.inputs_list[idx]["folder"].split("/")[-2]
                )

            input_data, label = self.normalize(input_data, label)

            return (
                input_data,
                torch.LongTensor(label),
                self.inputs_list[idx]["original_info"],
            )
        elif self.data_type == "lmdb":
            input_data, label, fi = self.read_lmdb(idx)
            input_data, label = self.normalize(input_data, label)
            return (
                input_data,
                torch.LongTensor(label),
                self.inputs_list[idx]["original_info"],
            )
        else:
            input_data, label = self.read_features(idx)
            return input_data, label, self.inputs_list[idx]["original_info"]

    def save_npz(self, video, label, idx):
        file_path = os.path.join(self.npz_folder, f"{idx}.npz")
        print(f"Novo .npz: {file_path}")
        np.savez_compressed(file_path, video=video, label=label)

    def read_video(self, index, num_glosses=-1):
        fi = self.inputs_list[index]
        img_folder = os.path.join(
            self.prefix, f"features/fullFrame-{IMG_SIZE}x{IMG_SIZE}px/" + fi["folder"]
        )
        img_list = sorted(glob.glob(img_folder))
        label_list = []
        for phase in fi["label"].split(" "):
            if phase == "":
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])
        return (
            [
                cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                for img_path in img_list
            ],
            label_list,
            fi,
        )

    def read_features(self, index):
        fi = self.inputs_list[index]
        data = np.load(
            f"{FEATURES_PATH}/{self.mode}/{fi['fileid']}_features.npy",
            allow_pickle=True,
        ).item()
        return data["features"], data["label"]

    def normalize(self, video, label, file_id=None):
        video, info = self.data_aug(video, label, None)

        if isinstance(info, dict):
            label = info.get("label", label)

        return video, label

    def to_tensor(self):
        return video_augmentation.Compose(
            [
                video_augmentation.ToTensor(),
            ]
        )

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.KorniaCompose(
                [
                    video_augmentation.ToList(),
                    video_augmentation.RandomCrop(AUGMENT_RANDOM_CROP),
                    video_augmentation.RandomHorizontalFlip(AUGMENT_HORIZONTAL_FLIP),
                    video_augmentation.ToTensor(),
                    video_augmentation.ChromaKeyVideo(mode=self.transform_mode),
                    video_augmentation.TemporalRescale(AUGMENT_TEMPORAL_RESCALE),
                    video_augmentation.Normalize(),
                    create_camera_simulation_pipeline(**self.kornia_kwargs),
                ]
            )
        else:
            print("Apply testing transform.")
            return video_augmentation.KorniaCompose(
                [
                    video_augmentation.ToList(),
                    video_augmentation.CenterCrop(AUGMENT_CENTER_CROP),
                    video_augmentation.ToTensor(),
                    video_augmentation.ChromaKeyVideo(mode="test"),
                    video_augmentation.Normalize(),
                ]
            )

    def byte_to_img(self, byteflow):
        unpacked = pa.deserialize(byteflow)
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        return img

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, info = list(zip(*batch))
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor(
                [np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video]
            )
            left_pad = 6
            right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
            max_len = max_len + left_pad + right_pad
            padded_video = [
                torch.cat(
                    (
                        vid[0][None].expand(left_pad, -1, -1, -1),
                        vid,
                        vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                    ),
                    dim=0,
                )
                for vid in video
            ]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [
                torch.cat(
                    (
                        vid,
                        vid[-1][None].expand(max_len - len(vid), -1),
                    ),
                    dim=0,
                )
                for vid in video
            ]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        label_length = torch.LongTensor([len(lab) for lab in label])
        if max(label_length) == 0:
            return padded_video, video_length, [], [], info
        else:
            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            padded_label = torch.LongTensor(padded_label)
            return padded_video, video_length, padded_label, label_length, info

    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


def process_data(data):
    return (data[2][0].split("|")[0], data[2][0].split("|")[-1])


def parallel_process(dataloader):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_data, data) for data in dataloader]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(result[0])
            print(result[-1])


if __name__ == "__main__":
    prefix = PREFIX
    gloss_dict = np.load(GLOSS_DICT_PATH, allow_pickle=True).item()
    feeder = BaseFeeder(prefix, gloss_dict)

    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=60,
    )

    parallel_process(dataloader)
