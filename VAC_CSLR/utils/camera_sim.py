import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import torch


class RandomBrightnessAdjustment(object):
    def __init__(self, brightness_factor=(0.8, 1.2)):
        self.brightness_factor = brightness_factor

    def __call__(self, clip):
        factor = random.uniform(self.brightness_factor[0], self.brightness_factor[1])
        clip = clip.float() * factor  # Convert to float for scaling
        return torch.clamp(clip, 0, 255).to(torch.uint8)


class RandomContrastAdjustment(object):
    def __init__(self, contrast_factor=(0.8, 1.2)):
        self.contrast_factor = contrast_factor

    def __call__(self, clip):
        clip = clip.float()  # Convert to float for mean calculation
        factor = random.uniform(self.contrast_factor[0], self.contrast_factor[1])
        mean = clip.mean(dim=(1, 2, 3), keepdim=True)
        clip = (clip - mean) * factor + mean
        return torch.clamp(clip, 0, 255).to(torch.uint8)


class RandomGaussianNoise(object):
    def __init__(self, mean=0, std_dev=10):
        self.mean = mean
        self.std_dev = std_dev

    def __call__(self, clip):
        noise = torch.randn_like(clip, dtype=torch.float32) * self.std_dev + self.mean
        clip = clip.float() + noise  # Add noise
        return torch.clamp(clip, 0, 255).to(torch.uint8)


class RandomSharpnessAdjustment(object):
    def __init__(self, sharpness_factor=(0.5, 1.5)):
        self.sharpness_factor = sharpness_factor

    def __call__(self, clip):
        factor = random.uniform(self.sharpness_factor[0], self.sharpness_factor[1])
        pil_clip = [Image.fromarray(img.permute(1, 2, 0).cpu().numpy()) for img in clip]
        adjusted = [torch.from_numpy(np.array(ImageEnhance.Sharpness(img).enhance(factor))) for img in pil_clip]
        return torch.stack([img.permute(2, 0, 1) for img in adjusted]).to(clip.dtype)


class RandomResolutionDownscale(object):
    def __init__(self, scale_range=(0.5, 1.0)):
        self.scale_range = scale_range

    def __call__(self, clip):
        scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
        downscaled = []
        for img in clip:
            h, w = img.shape[1:]
            new_size = (int(w * scale_factor), int(h * scale_factor))
            small_img = cv2.resize(img.permute(1, 2, 0).cpu().numpy(), new_size, interpolation=cv2.INTER_LINEAR)
            upscaled_img = cv2.resize(small_img, (w, h), interpolation=cv2.INTER_LINEAR)
            downscaled.append(torch.from_numpy(upscaled_img).permute(2, 0, 1))
        return torch.stack(downscaled).to(clip.dtype)


class RandomPerspectiveTransform(object):
    def __init__(self, max_shift=10):
        self.max_shift = max_shift

    def __call__(self, clip):
        h, w = clip.shape[2:]
        points1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        points2 = points1 + np.random.uniform(-self.max_shift, self.max_shift, points1.shape).astype(np.float32)
        M = cv2.getPerspectiveTransform(points1, points2)
        transformed = [cv2.warpPerspective(img.permute(1, 2, 0).cpu().numpy(), M, (w, h)) for img in clip]
        return torch.stack([torch.from_numpy(img).permute(2, 0, 1) for img in transformed]).to(clip.dtype)


class RandomPercentileEqualization(object):
    def __init__(self, percentile_range=(0.85, 1)):
        """
        Args:
            percentile_range (tuple): Range of percentiles to sample from.
        """
        self.percentile_range = percentile_range

    def __call__(self, clip):
        """
        Applies histogram equalization based on a chosen percentile.

        Args:
            clip (torch.Tensor): Video tensor of shape (T, C, H, W) with values in [0, 255].

        Returns:
            torch.Tensor: Equalized video tensor in [0, 255].
        """
        if not isinstance(clip, torch.Tensor):
            raise TypeError("Expected clip to be a torch.Tensor")

        clip = clip.float()  # Ensure float for calculations

        # Randomly select a percentile from the given range
        chosen_percentile = random.uniform(self.percentile_range[0], self.percentile_range[1])

        # Keep compatibility with older PyTorch versions that do not support
        # the interpolation keyword in torch.quantile.
        flattened = clip.view(clip.shape[0], clip.shape[1], -1)
        try:
            percentile_values = torch.quantile(
                flattened,
                chosen_percentile,
                dim=-1,
                keepdim=True,
                interpolation="nearest",
            )
        except TypeError:
            percentile_values = torch.quantile(
                flattened,
                chosen_percentile,
                dim=-1,
                keepdim=True,
            )
        percentile_values = percentile_values.unsqueeze(-1)

        # Clamp with tensor bounds in a way compatible with older PyTorch.
        clip_floor = clip.amin(dim=(2, 3), keepdim=True)
        clip = torch.max(clip, clip_floor)
        clip = torch.min(clip, percentile_values)

        # Normalize using histogram equalization principles
        clip_min = clip.amin(dim=(2, 3), keepdim=True)  # Min value of any frame in the video
        clip_max = clip.amax(dim=(2, 3), keepdim=True)  # Max value of any frame in the video

        # Avoid division by zero by ensuring that the difference is not too small
        clip_range = torch.clamp(clip_max - clip_min, min=1e-6)
        clip = 255 * (clip - clip_min) / clip_range  # Normalize the clip

        return clip


class CameraSimulationPipeline(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        if not isinstance(clip, torch.Tensor):
            raise TypeError("Expected clip to be a torch.Tensor")
        for transform in self.transforms:
            clip = transform(clip)
        return clip


def create_camera_simulation_pipeline(**kwargs):
    transforms = []
    if percentile_equalization := kwargs.get("percentile_equalization"):
        transforms.append(RandomPercentileEqualization(**percentile_equalization))
    if brightness := kwargs.get("brightness"):
        transforms.append(RandomBrightnessAdjustment(**brightness))
    if contrast := kwargs.get("contrast"):
        transforms.append(RandomContrastAdjustment(**contrast))
    if gaussian_noise := kwargs.get("gaussian_noise"):
        transforms.append(RandomGaussianNoise(**gaussian_noise))
    if sharpness := kwargs.get("sharpness"):
        transforms.append(RandomSharpnessAdjustment(**sharpness))
    if resolution_downscale := kwargs.get("resolution_downscale"):
        transforms.append(RandomResolutionDownscale(**resolution_downscale))
    if perspective_transform := kwargs.get("perspective_transform"):
        transforms.append(RandomPerspectiveTransform(**perspective_transform))
    return CameraSimulationPipeline(transforms)
