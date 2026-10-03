import torch
import torch.nn as nn
import kornia.augmentation as K
import kornia.augmentation.container as KContainer
from kornia.geometry.transform import Resize
import numpy as np
from augmentation.video.utils.custom_transformations import ChromaKeyVideo


class Transformation(nn.Module):
    def __init__(
        self,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        strength=1.0,
        random_shift=4,
        stride=1,
        max_seq_len=512,
    ):
        super().__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

        self.random_shift = random_shift
        self.stride = stride
        self.max_seq_len = max_seq_len
        s = strength

        self.chroma_key_t = ChromaKeyVideo(tola=40, tolb=40, mode="train")

        self.chroma_key_v = ChromaKeyVideo(tola=40, tolb=40, mode="test")

        self.transform_train = KContainer.ImageSequential(
            K.RandomRotation(
                degrees=5.0,
                p=0.3,
                same_on_batch=True,
            ),
            K.RandomResizedCrop(
                size=(224, 224),
                scale=(0.875, 1.0),
                ratio=(0.9, 1.1),
                p=1.0,
                same_on_batch=True,
            ),
            K.RandomHorizontalFlip(
                p=0.5,
                same_on_batch=True,
            ),
            K.ColorJitter(
                brightness=0.8 * s,
                contrast=0.8 * s,
                saturation=0.8 * s,
                hue=0.2 * s,
                p=0.3,
                same_on_batch=True,
            ),
            K.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                p=0.5,
                same_on_batch=True,
                keepdim=True,
            ),
            K.RandomBrightness(
                brightness=(0.3, 1.7),
                p=0.5,
                same_on_batch=True,
                clip_output=True,
                keepdim=True,
            ),
            K.RandomContrast(
                contrast=(0.5, 1.5),
                p=0.5,
                same_on_batch=True,
                clip_output=True,
                keepdim=True,
            ),
            K.RandomGaussianNoise(
                mean=0.0, std=0.2, p=0.5, same_on_batch=True, keepdim=True
            ),
            K.RandomPerspective(
                distortion_scale=0.2, p=0.1, same_on_batch=True, keepdim=True
            ),
            K.RandomPlanckianJitter(
                mode="blackbody", p=0.05, same_on_batch=True, keepdim=True
            ),
            K.RandomPlasmaBrightness(
                intensity=(0.0, 0.6),
                roughness=(0.1, 0.7),
                p=0.1,
                same_on_batch=True,
                keepdim=True,
            ),
            K.RandomSharpness(
                sharpness=(0.4, 1.6), p=0.5, same_on_batch=True, keepdim=True
            ),
            same_on_batch=True,
        ).to(self.device)

        self.transform_valid = KContainer.ImageSequential(
            Resize((224, 224)), same_on_batch=True
        ).to(self.device)

        self.shift_factor = 0.1
        self.scale_factor = 0.125
        self.normalize = K.Normalize(
            mean=self.mean, std=self.std).to(self.device)
        self.unnorm = K.Denormalize(
            mean=self.mean, std=self.std).to(self.device)

    def aug_video(self, frames, isValid):
        frames = [
            torch.from_numpy(f).permute(
                2, 0, 1) if isinstance(f, np.ndarray) else f
            for f in frames
        ]
        frames = torch.stack(frames)

        if isValid:
            frames = self.chroma_key_v(
                frames)
        else:
            frames = self.chroma_key_t(frames)

        frames = frames.float().div(255.0).to(self.device)

        if isValid:
            frames = self.transform_valid(frames)
        else:
            frames = self.transform_train(frames)

        frames = self.normalize(frames)

        return frames

    def aug_kpt(self, poses, isValid):
        if not isValid:
            shift_values = torch.randn(2) * self.shift_factor
            scale_values = torch.FloatTensor(1).uniform_(
                1.0 - self.scale_factor, 1.0 + self.scale_factor
            )

            for k, pose in poses.items():
                poses[k][:, :, :2] = (
                    pose[:, :, :2] + shift_values) * scale_values

        for k, pose in poses.items():
            poses[k][:, :, :2] = poses[k][:, :, :2] * 2 - 1

        return poses

    def unnorm_imgs(self, frames):
        output = []
        for frame in frames:
            output.append(self.unnorm(frame).detach().cpu().numpy())
        return output

    def to(self, device):
        super().to(device)
        self.transform_train.to(device)
        self.transform_valid.to(device)
        return self
