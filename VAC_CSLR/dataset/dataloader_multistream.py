import importlib
import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from utils import camera_sim
from utils import video_augmentation

_base = importlib.import_module("dataset.dataloader_gn_chroma_int_npz_cam-sim_phfps")
BaseFeeder = _base.BaseFeeder
TORCH_MEAN = _base.TORCH_MEAN
TORCH_STD = _base.TORCH_STD
TRIM_CSV_PATH = _base.TRIM_CSV_PATH
AUGMENT_HORIZONTAL_FLIP = _base.AUGMENT_HORIZONTAL_FLIP
AUGMENT_TEMPORAL_RESCALE = _base.AUGMENT_TEMPORAL_RESCALE
AUGMENT_CENTER_CROP = _base.AUGMENT_CENTER_CROP

AUX_NPZ_FOLDER = os.getenv("AUX_NPZ_FOLDER")
DEPTH_MAX_MM = float(os.getenv("DEPTH_MAX_MM"))


class MultiStreamFeeder(BaseFeeder):
    valid_indices = None

    def __init__(self, *args, aux_streams, **kwargs):
        if "depth" in aux_streams and any(s != "depth" for s in aux_streams):
            raise ValueError(f"Cannot mix depth with RGB streams: {aux_streams}")
        self.aux_streams = list(aux_streams)
        self.is_depth = "depth" in self.aux_streams
        super().__init__(*args, **kwargs)
        keep = [i for i in range(len(self.inputs_list) - 1) if self.aux_exists(i)]
        print(f"{self.mode}: dropped {len(self.inputs_list) - 1 - len(keep)} of "
              f"{len(self.inputs_list) - 1} samples with missing aux files {self.aux_streams}")
        self.valid_indices = keep
        print(self.mode, len(self))

    def transform(self):
        # shared temporal / spatial ops (applied to every stream) and per-stream photometric ops
        self.to_tensor_op = video_augmentation.ToTensor()
        self.half_fps = video_augmentation.ProbabilisticHalfFps()
        self.normalize_rgb = transforms.Normalize(mean=TORCH_MEAN, std=TORCH_STD)
        if self.transform_mode == "train":
            print("Apply training transform.")
            self.clip_trim = video_augmentation.ClipTrim(TRIM_CSV_PATH)
            self.temporal_rescale = video_augmentation.TemporalRescale(AUGMENT_TEMPORAL_RESCALE)
            self.resize = video_augmentation.Resize(224 / 256)
            self.chroma = video_augmentation.ChromaKeyVideo(tola=40, tolb=40, mode=self.transform_mode)
            self.cam_sim = camera_sim.create_camera_simulation_pipeline(**self.cam_sim_kwargs)
        else:
            print("Apply testing transform.")
            self.center_crop = video_augmentation.CenterCrop(AUGMENT_CENTER_CROP)
            self.chroma = video_augmentation.ChromaKeyVideo(mode="test")
        return None

    def aux_path(self, stream, idx):
        video_id = self.inputs_list[idx]["folder"].split("/")[-2]
        return f"{AUX_NPZ_FOLDER}{stream}/{self.mode}/{video_id}.npz"

    def aux_exists(self, idx):
        return all(os.path.exists(self.aux_path(s, idx)) for s in self.aux_streams)

    def __len__(self):
        if self.valid_indices is None:
            return super().__len__()
        return len(self.valid_indices)

    def load_frontal(self, idx):
        video_id = self.inputs_list[idx]["folder"].split("/")[-2]
        file_path = os.path.join(self.npz_folder, f"{video_id}.npz")
        input_data = None
        if os.path.exists(file_path):
            try:
                data = np.load(file_path)
                input_data = data["video"]
                label = data["label"].tolist()
            except Exception as e:
                print(f"Erro ao carregar o arquivo {file_path}: {e}")
        if input_data is None:
            input_data, label, _ = self.read_video(idx)
            input_data = np.array(input_data)
            self.save_npz(input_data, label, video_id)
        return input_data, label  # T x H x W x 3 uint8

    def load_aux(self, stream, idx, length):
        aux = np.load(self.aux_path(stream, idx))["video"]  # T_a x H x W (x 3)
        return aux[np.round(np.linspace(0, len(aux) - 1, length)).astype(int)]  # T x H x W (x 3)

    def spatial(self, clip, flip, depth=False):
        # clip: T x H x W x 3 uint8 (RGB) or T x H x W uint16 (depth)
        if self.transform_mode == "train":
            if depth:
                h, w = clip.shape[1:3]
                clip = np.stack([cv2.resize(f, (int(w * 224 / 256), int(h * 224 / 256)),
                                            interpolation=cv2.INTER_NEAREST) for f in clip])  # T x H' x W'
            else:
                clip = np.stack(self.resize(clip))  # T x H' x W' x 3
            if flip:
                clip = np.ascontiguousarray(clip[:, :, ::-1])
        else:
            # CenterCrop expects frames with a channel axis
            clip = np.stack(self.center_crop(clip[..., None] if depth else clip))
        return clip[..., None] if depth and clip.ndim == 3 else clip  # T x H x W x C

    def photometric_rgb(self, clip):
        clip = self.to_tensor_op(clip)  # T x 3 x H x W float [0, 255]
        clip = self.chroma(clip)  # T x 3 x H x W
        if self.transform_mode == "train":
            clip = self.cam_sim(clip)  # T x 3 x H x W
        return self.normalize_rgb(clip.float() / 255)  # T x 3 x H x W

    def normalize_depth(self, clip):
        clip = torch.from_numpy(clip.astype(np.float32))  # T x H x W x 1
        clip = clip.clamp(0, DEPTH_MAX_MM) / DEPTH_MAX_MM
        return ((clip - 0.5) / 0.5).permute(0, 3, 1, 2)  # T x 1 x H x W

    def __getitem__(self, idx):
        idx = self.valid_indices[idx]
        item = self.inputs_list[idx]
        video, label = self.load_frontal(idx)  # T x H x W x 3
        length = len(video)
        auxs = [self.load_aux(s, idx, length) for s in self.aux_streams]  # V x [T x H x W (x 3)]

        # one temporal index shared by all streams
        index = np.arange(length)
        if self.transform_mode == "train":
            index = self.clip_trim(index, item["fileid"])
        index = self.half_fps(index)
        if self.transform_mode == "train":
            index = self.temporal_rescale(index)

        flip = self.transform_mode == "train" and random.random() < AUGMENT_HORIZONTAL_FLIP
        video = self.photometric_rgb(self.spatial(video[index], flip))  # T x 3 x H x W
        if self.is_depth:
            auxs = [self.normalize_depth(self.spatial(a[index], flip, depth=True)) for a in auxs]  # V x [T x 1 x H x W]
        else:
            auxs = [self.photometric_rgb(self.spatial(a[index], flip)) for a in auxs]  # V x [T x 3 x H x W]
        aux = torch.stack(auxs, dim=1)  # T x V x C x H x W
        return video, torch.LongTensor(label), item["original_info"], aux

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, info, aux = list(zip(*batch))
        max_len = len(video[0])
        video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video])
        left_pad = 6
        right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
        max_len = max_len + left_pad + right_pad

        def pad(clip):
            return torch.cat(
                (
                    clip[0][None].expand(left_pad, *clip.shape[1:]),
                    clip,
                    clip[-1][None].expand(max_len - len(clip) - left_pad, *clip.shape[1:]),
                ),
                dim=0,
            )

        padded_video = torch.stack([pad(vid) for vid in video])  # B x T x 3 x H x W
        padded_aux = torch.stack([pad(a) for a in aux])  # B x T x V x C x H x W
        aux_length = video_length.clone()
        label_length = torch.LongTensor([len(lab) for lab in label])
        if max(label_length) == 0:
            return padded_video, video_length, [], [], padded_aux, aux_length, info
        padded_label = []
        for lab in label:
            padded_label.extend(lab)
        padded_label = torch.LongTensor(padded_label)
        return padded_video, video_length, padded_label, label_length, padded_aux, aux_length, info
