import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")
from kornia.augmentation import (  # noqa: E402
    RandomMotionBlur,
    RandomPlanckianJitter,
    RandomPlasmaBrightness,
    RandomBrightness,
    RandomContrast,
    RandomGaussianNoise,
    RandomSharpness,
    RandomAffine,
    RandomPerspective
)


class ExtendedCameraSimulationPipeline:
    def __init__(self, **kwargs):
        self.transforms = []
        if brightness := kwargs.get("brightness"):
            self.transforms.append(RandomBrightness(**brightness))
        if contrast := kwargs.get("contrast"):
            self.transforms.append(RandomContrast(**contrast))
        if gaussian_noise := kwargs.get("gaussian_noise"):
            self.transforms.append(RandomGaussianNoise(**gaussian_noise))
        if sharpness := kwargs.get("sharpness"):
            self.transforms.append(RandomSharpness(**sharpness))
        if motion_blur := kwargs.get("motion_blur"):
            self.transforms.append(RandomMotionBlur(**motion_blur))
        if planckian_jitter := kwargs.get("planckian_jitter"):
            self.transforms.append(RandomPlanckianJitter(**planckian_jitter))
        if plasma_brightness := kwargs.get("plasma_brightness"):
            self.transforms.append(RandomPlasmaBrightness(**plasma_brightness))
        if affine := kwargs.get("affine"):
            self.transforms.append(RandomAffine(**affine))
        if perspective := kwargs.get("perspective"):
            self.transforms.append(RandomPerspective(**perspective))
            
    def __call__(self, video):
        if video.type() != "torch.FloatTensor":
            video = video.float()
        
        if video.max() > 1.0:
            video = video / 255.0
            
        for transform in self.transforms:
            video = transform(video.contiguous())
            
        video = video * 255.0
        
        return video


def create_camera_simulation_pipeline(**kwargs):
    return ExtendedCameraSimulationPipeline(**kwargs)
