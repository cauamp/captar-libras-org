import torch
from augmentation.video.utils.unnorm import UnNormalize
import augmentation.video.utils.custom_transformations as aug
import augmentation.video.utils.camera_sim as camera_sim


class Transformation(object):
    def __init__(
        self,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        strength=0.2,
        random_shift=4,
        stride=1,
        max_seq_len=512,
        cam_sim_kwargs={}
    ):
        self.mean = mean
        self.std = std

        self.random_shift = random_shift
        self.stride = stride
        self.max_seq_len = max_seq_len

        self.cam_sim_kwargs = cam_sim_kwargs

        self.transform_train = aug.Compose(
            [
                aug.ToList(),
                aug.ProbabilisticHalfFps(),
                aug.Resize(224/256),
                aug.RandomHorizontalFlip(0.5),
                aug.ToTensor(),
                aug.ChromaKeyVideo(tola=40, tolb=40, mode="train"),
                aug.TemporalRescale(strength),
                camera_sim.create_camera_simulation_pipeline(
                    **self.cam_sim_kwargs),
            ]
        )

        self.transform_valid = aug.Compose(
            [
                aug.ToList(),
                aug.ProbabilisticHalfFps(),
                aug.CenterCrop(224),
                aug.ToTensor(),
                aug.ChromaKeyVideo(mode="test"),
            ]
        )

        self.shift_factor = 0.1
        self.scale_factor = 0.125
        self.unnorm = UnNormalize(mean, std)

    def aug_video(self, frames, isValid):
        if isValid:
            transformed_frames = self.transform_valid(frames)
        else:
            transformed_frames = self.transform_train(frames)

        transformed_frames = transformed_frames.float() / 255.0
        mean = torch.tensor(self.mean).view(1, 3, 1, 1)
        std = torch.tensor(self.std).view(1, 3, 1, 1)
        transformed_frames = (transformed_frames - mean) / std

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
