import torch
from augmentation.video.utils.unnorm import UnNormalize
from augmentation.video.utils import video_augmentation


class Transformation(object):
    def __init__(
        self,
        individual_normalization=False,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        mode="yolo_hf",
    ):
        self.individual_normalization = individual_normalization
        self.mean = mean
        self.std = std

        if mode == "yolo":
            print("Applying YoloBBCrop")

            self.transform = video_augmentation.Compose([
                video_augmentation.YoloBBCrop(224),
                video_augmentation.ToTensor(),
            ])

        elif mode == "yolo_hf":
            print("Applying YoloBBCrop With 15 fps")
            self.transform = video_augmentation.Compose([
                video_augmentation.HalfFps(),
                video_augmentation.YoloBBCrop(224),
                video_augmentation.ToTensor(),
            ])
        else:
            self.transform = video_augmentation.Compose([
                video_augmentation.CenterCrop(224),
                video_augmentation.ToTensor(),
            ])

        self.shift_factor = 0.1
        self.scale_factor = 0.125

    def aug_video(self, frames, isValid):

        transformed_frames = self.transform(frames)

        transformed_frames = transformed_frames.float() / 255.0

        # Individual normalization
        if self.individual_normalization:
            self.mean = transformed_frames.mean(dim=(0, 2, 3), keepdim=True)
            self.std = transformed_frames.std(dim=(0, 2, 3), keepdim=True)

        mean = torch.tensor(self.mean).view(1, 3, 1, 1)
        std = torch.tensor(self.std).view(1, 3, 1, 1)
        transformed_frames = (transformed_frames - mean) / std
        self.unnorm = UnNormalize(mean, std)

        return transformed_frames

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
