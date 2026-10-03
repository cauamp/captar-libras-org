# ----------------------------------------
# Written by Yuecong Min
# ----------------------------------------
import cv2
import PIL
import copy
import scipy.misc
import torch
import random
import numbers
import numpy as np
import torchvision.transforms as transforms
import glob
from ultralytics import YOLO
import logging
import csv

logging.getLogger("ultralytics").setLevel(logging.WARNING)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label, file_info=None):
        for t in self.transforms:
            if file_info is not None and isinstance(t, WERAugment):
                image, label = t(image, label, file_info)
            elif file_info is not None and isinstance(t, ClipTrim):
                image = t(image, file_info)
            else:
                image = t(image)
        return image, label


class KorniaCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label, file_info=None):
        info = {}
        for t in self.transforms:
            if file_info is not None and isinstance(t, WERAugment):
                image, label = t(image, label, file_info)
                info["label"] = label
            if isinstance(t, Normalize):
                image, means, std = t(image)
                info["means"] = means
                info["stds"] = std
            else:
                image = t(image)
        return image, info


class WERAugment(object):
    def __init__(self, boundary_path):
        self.boundary_dict = np.load(boundary_path, allow_pickle=True).item()
        self.K = 3

    def __call__(self, video, label, file_info):
        ind = np.arange(len(video)).tolist()
        if file_info not in self.boundary_dict.keys():
            return video, label
        binfo = copy.deepcopy(self.boundary_dict[file_info])
        binfo = [0] + binfo + [len(video)]
        k = np.random.randint(min(self.K, len(label) - 1))
        for i in range(k):
            ind, label, binfo = self.one_operation(ind, label, binfo)
        ret_video = [video[i] for i in ind]
        return ret_video, label

    def one_operation(self, *inputs):
        prob = np.random.random()
        if prob < 0.3:
            return self.delete(*inputs)
        elif 0.3 <= prob < 0.7:
            return self.substitute(*inputs)
        else:
            return self.insert(*inputs)

    @staticmethod
    def delete(ind, label, binfo):
        del_wd = np.random.randint(len(label))
        ind = ind[: binfo[del_wd]] + ind[binfo[del_wd + 1] :]
        duration = binfo[del_wd + 1] - binfo[del_wd]
        del label[del_wd]
        binfo = [i for i in binfo[:del_wd]] + [i - duration for i in binfo[del_wd + 1 :]]
        return ind, label, binfo

    @staticmethod
    def insert(ind, label, binfo):
        ins_wd = np.random.randint(len(label))
        ins_pos = np.random.choice(binfo)
        ins_lab_pos = binfo.index(ins_pos)

        ind = ind[:ins_pos] + ind[binfo[ins_wd] : binfo[ins_wd + 1]] + ind[ins_pos:]
        duration = binfo[ins_wd + 1] - binfo[ins_wd]
        label = label[:ins_lab_pos] + [label[ins_wd]] + label[ins_lab_pos:]
        binfo = binfo[:ins_lab_pos] + [binfo[ins_lab_pos - 1] + duration] + [i + duration for i in binfo[ins_lab_pos:]]
        return ind, label, binfo

    @staticmethod
    def substitute(ind, label, binfo):
        sub_wd = np.random.randint(len(label))
        tar_wd = np.random.randint(len(label))

        ind = ind[: binfo[tar_wd]] + ind[binfo[sub_wd] : binfo[sub_wd + 1]] + ind[binfo[tar_wd + 1] :]
        label[tar_wd] = label[sub_wd]
        delta_duration = binfo[sub_wd + 1] - binfo[sub_wd] - (binfo[tar_wd + 1] - binfo[tar_wd])
        binfo = binfo[: tar_wd + 1] + [i + delta_duration for i in binfo[tar_wd + 1 :]]
        return ind, label, binfo


class ToTensor(object):
    def __call__(self, video):
        if isinstance(video, list):
            video = np.array(video)
            video = torch.from_numpy(video.transpose((0, 3, 1, 2))).float()
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video.transpose((0, 3, 1, 2))).float()
        return video


class ToList(object):
    def __call__(self, video):
        if isinstance(video, list):
            return video

        if isinstance(video, torch.Tensor):
            video = video.permute(0, 2, 3, 1).cpu().numpy()
            video_list = [np.array(item) for item in video]
        if isinstance(video, np.ndarray):
            if video.ndim == 4:  # Verifica se o tensor tem 4 dimensões
                video_list = [np.array(item) for item in video]
            else:
                raise ValueError("O numpy.ndarray deve ter 4 dimensões: (N, altura, largura, canais)")
        return video_list


class ToNumpyArray(object):
    def __call__(self, video):
        if isinstance(video, torch.Tensor):
            video = video.permute(0, 2, 3, 1).cpu().numpy()
        return video


class RandomCrop(object):
    """
    Extract random crop of the video.
    Args:
        size (sequence or int): Desired output size for the crop in format (h, w).
        crop_position (str): Selected corner (or center) position from the
        list ['c', 'tl', 'tr', 'bl', 'br']. If it is non, crop position is
        selected randomly at each call.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError("If size is a single number, it must be positive")
            size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError("If size is a sequence, it must be of len 2.")
        self.size = size

    def __call__(self, clip):
        crop_h, crop_w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError("Expected numpy.ndarray or PIL.Image" + "but got list of {0}".format(type(clip[0])))
        if crop_w > im_w:
            pad = crop_w - im_w
            clip = [
                np.pad(img, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)), "constant", constant_values=0) for img in clip
            ]
            w1 = 0
        else:
            w1 = random.randint(0, im_w - crop_w)

        if crop_h > im_h:
            pad = crop_h - im_h
            clip = [
                np.pad(img, ((pad // 2, pad - pad // 2), (0, 0), (0, 0)), "constant", constant_values=0) for img in clip
            ]
            h1 = 0
        else:
            h1 = random.randint(0, im_h - crop_h)

        if isinstance(clip[0], np.ndarray):
            return [img[h1 : h1 + crop_h, w1 : w1 + crop_w, :] for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.crop((w1, h1, w1 + crop_w, h1 + crop_h)) for img in clip]


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        try:
            im_h, im_w, im_c = clip[0].shape
        except ValueError:
            print(clip[0].shape)
        new_h, new_w = self.size
        new_h = im_h if new_h >= im_h else new_h
        new_w = im_w if new_w >= im_w else new_w
        top = int(round((im_h - new_h) / 2.0))
        left = int(round((im_w - new_w) / 2.0))
        return [img[top : top + new_h, left : left + new_w] for img in clip]


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, clip):
        # B, H, W, 3
        flag = random.random() < self.prob
        if flag:
            clip = np.flip(clip, axis=2)
            clip = np.ascontiguousarray(copy.deepcopy(clip))
        return np.array(clip)


class RandomRotation(object):
    """
    Rotate entire clip randomly by a random angle within
    given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number,must be positive")
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence,it must be of len 2.")
        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [scipy.misc.imrotate(img, angle) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError("Expected numpy.ndarray or PIL.Image" + "but got list of {0}".format(type(clip[0])))
        return rotated


class TemporalRescale(object):
    def __init__(self, temp_scaling=0.2):
        self.min_len = 32
        self.max_len = 230
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling

    def __call__(self, clip):
        vid_len = len(clip)
        new_len = int(vid_len * (self.L + (self.U - self.L) * np.random.random()))
        if new_len < self.min_len:
            new_len = self.min_len
        if new_len > self.max_len:
            new_len = self.max_len
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        if new_len <= vid_len:
            index = sorted(random.sample(range(vid_len), new_len))
        else:
            index = sorted(random.choices(range(vid_len), k=new_len))
        return clip[index]


class RandomResize(object):
    """
    Resize video bysoomingin and out.
    Args:
        rate (float): Video is scaled uniformly between
        [1 - rate, 1 + rate].
        interp (string): Interpolation to use for re-sizing
        ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic').
    """

    def __init__(self, rate=0.0, interp="bilinear"):
        self.rate = rate
        self.interpolation = interp

    def __call__(self, clip):
        scaling_factor = random.uniform(1 - self.rate, 1 + self.rate)

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_h, new_w)
        if isinstance(clip[0], np.ndarray):
            return [scipy.misc.imresize(img, size=(new_h, new_w), interp=self.interpolation) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.resize(size=(new_w, new_h), resample=self._get_PIL_interp(self.interpolation)) for img in clip]
        else:
            raise TypeError("Expected numpy.ndarray or PIL.Image" + "but got list of {0}".format(type(clip[0])))

    def _get_PIL_interp(self, interp):
        if interp == "nearest":
            return PIL.Image.NEAREST
        elif interp == "lanczos":
            return PIL.Image.LANCZOS
        elif interp == "bilinear":
            return PIL.Image.BILINEAR
        elif interp == "bicubic":
            return PIL.Image.BICUBIC
        elif interp == "cubic":
            return PIL.Image.CUBIC


class Resize(object):
    """
    Resize video bysoomingin and out.
    Args:
        rate (float): Video is scaled uniformly between
        [1 - rate, 1 + rate].
        interp (string): Interpolation to use for re-sizing
        ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic').
    """

    def __init__(self, rate=0.0, interp="bilinear"):
        self.rate = rate
        self.interpolation = interp

    def __call__(self, clip):
        scaling_factor = self.rate

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)
        if isinstance(clip[0], np.ndarray):
            return [np.array(PIL.Image.fromarray(img).resize(new_size)) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.resize(size=(new_w, new_h), resample=self._get_PIL_interp(self.interpolation)) for img in clip]
        else:
            raise TypeError("Expected numpy.ndarray or PIL.Image" + "but got list of {0}".format(type(clip[0])))

    def _get_PIL_interp(self, interp):
        if interp == "nearest":
            return PIL.Image.NEAREST
        elif interp == "lanczos":
            return PIL.Image.LANCZOS
        elif interp == "bilinear":
            return PIL.Image.BILINEAR
        elif interp == "bicubic":
            return PIL.Image.BICUBIC
        elif interp == "cubic":
            return PIL.Image.CUBIC


class RandomColorZScoreJitterVideo:
    def __init__(
        self, brightness_range=(1.2, 1.4), contrast_range=(1.2, 1.3), saturation_range=(1.2, 1.5), hue_range=(0.0, 0.0)
    ):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.set_params()

    def set_params(self):
        brightness = random.uniform(*self.brightness_range)
        contrast = random.uniform(*self.contrast_range)
        saturation = random.uniform(*self.saturation_range)
        hue = random.uniform(*self.hue_range)

        self.transform = transforms.ColorJitter(
            brightness=(brightness, brightness),
            contrast=(contrast, contrast),
            saturation=(saturation, saturation),
            hue=(hue, hue),
        )

    def clamp_values(self, tensor, min_value, max_value):
        return torch.clamp(tensor, min_value, max_value)

    def __call__(self, clip):
        if self.transform is None:
            self.set_params()

        if random.uniform(0, 1) < 0.5:
            return clip

        clip = clip.to("cpu")

        normalized_clip = (clip + 1) / 2
        normalized_clip = self.clamp_values(normalized_clip, 0.0, 255.0)

        jittered_clip = torch.stack([self.transform(frame) for frame in normalized_clip])

        jittered_clip = self.clamp_values(jittered_clip, 0.0, 1.0)
        jittered_clip = (jittered_clip * 2) - 1
        jittered_clip = self.clamp_values(jittered_clip, -1.0, 1.0)

        return jittered_clip


class RandomColorJitterVideo:
    def __init__(
        self, brightness_range=(1.2, 1.5), contrast_range=(1.2, 1.3), saturation_range=(1.2, 1.5), hue_range=(0.0, 0.0)
    ):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.set_params()

    def set_params(self):
        brightness = random.uniform(*self.brightness_range)
        contrast = random.uniform(*self.contrast_range)
        saturation = random.uniform(*self.saturation_range)
        hue = random.uniform(*self.hue_range)

        self.transform = transforms.ColorJitter(
            brightness=(brightness, brightness),
            contrast=(contrast, contrast),
            saturation=(saturation, saturation),
            hue=(hue, hue),
        )

    def __call__(self, clip):
        if self.transform is None:
            self.set_params()

        if random.uniform(0, 1) < 0.5:
            return clip

        jittered_clip = torch.stack([self.transform(frame) for frame in clip])

        return jittered_clip


class ChromaKeyVideo:
    def __init__(
        self, key_color=(10, 130, 60), tola=35, tolb=35, backgrounds_path="./datasets_srv/backgrounds", mode="train"
    ):
        # Define as cores chave como RGB
        self.key_color = torch.tensor(key_color).view(3, 1, 1).float()  # Formata para tensor 3x1x1
        self.tola = tola
        self.tolb = tolb
        self.backgrounds_list = (
            glob.glob(backgrounds_path + f"/{mode}/*.jpg") + glob.glob(backgrounds_path + f"/{mode}/*.png") + [None]
        )

    def __call__(self, clip: torch.Tensor):
        """
        Aplica o chroma key em cada frame do vídeo e substitui com o fundo.

        Parâmetros:
        - clip: Vídeo de entrada como um tensor [T, C, H, W] (T = nº de frames, C = nº de canais, H = altura, W = largura).

        Retorna:
        - result_clip: Vídeo com chroma key aplicado como tensor [T, C, H, W].
        """

        if not isinstance(clip[0], torch.Tensor):
            raise TypeError(
                f"At {self.__class__} Expected list of torch.Tensor [0, 255]"
                + "but got list of {0}".format(type(clip[0]))
            )

        # Carregar a imagem de fundo
        background = self.load_background_image(random.choice(self.backgrounds_list), clip.shape[2:], clip.shape[0])

        if background is None:
            return clip

        # Verificar se o tamanho do foreground e do background são compatíveis
        if clip.shape != background.shape:
            raise ValueError("O foreground e o background devem ter o mesmo tamanho.")

        # Aplicar o chroma key a cada frame do vídeo
        result_clip = torch.stack(
            [self.apply_chroma_key(frame, background_frame) for frame, background_frame in zip(clip, background)]
        )
        return result_clip

    def load_background_image(self, image_path: str, frame_size: list, num_frames: int):
        """
        Carrega a imagem de fundo de um arquivo .png e a ajusta ao tamanho dos frames do vídeo.

        Parâmetros:
        - image_path: Caminho para o arquivo de imagem (.png).
        - frame_size: Tupla (H, W) representando a altura e largura dos frames do vídeo.

        Retorna:
        - background: Tensor [C, H, W] com a imagem redimensionada para ser usada como fundo.
        """
        if image_path is None:
            return None

        # Abrir a imagem de fundo usando PIL
        background_image = PIL.Image.open(image_path).convert("RGB")

        # Redimensionar a imagem para corresponder ao tamanho dos frames
        # PIL usa (W, H), enquanto frame usa (H, W)
        background_image = background_image.resize(frame_size[::-1])

        # Converter a imagem para tensor (C, H, W) no intervalo [0, 1]
        transform_to_tensor = transforms.ToTensor()
        background = transform_to_tensor(background_image) * 255

        # Expande a dimensão para ser um batch com um frame
        return background.unsqueeze(0).repeat(num_frames, 1, 1, 1)

    def create_mask(self, foreground_ycbcr: float, keycolor_ycbcr: float):
        # Calcula a distância entre os canais Y, Cb e Cr
        dist = np.sqrt(
            (foreground_ycbcr[:, :, 1] - keycolor_ycbcr[1]) ** 2 + (foreground_ycbcr[:, :, 2] - keycolor_ycbcr[2]) ** 2
        )

        # Cria a máscara com base nas tolerâncias
        mask = np.zeros_like(dist, dtype=np.float32)
        mask[dist > self.tola] = 0
        mask[(dist >= self.tola) & (dist < self.tolb)] = (
            dist[(dist >= self.tola) & (dist < self.tolb)] - self.tola
        ) / (self.tolb - self.tola)
        mask[dist >= self.tolb] = 1
        return 1 - mask

    def apply_chroma_key(self, foreground: torch.FloatTensor, background: torch.FloatTensor):
        """
        Aplica o chroma key a um frame individual.

        Parâmetros:
        - foreground: Tensor [C, H, W] com a imagem de primeiro plano (foreground).
        - background: Tensor [C, H, W] com a imagem de fundo (background).

        Retorna:
        - frame_result: Tensor [C, H, W] com o chroma key aplicado.
        """

        # Converte o foreground para YCbCr
        # De [C, H, W] para [H, W, C]
        foreground_image = (foreground.permute(1, 2, 0) * 1).byte()
        foreground_ycbcr = PIL.Image.fromarray(foreground_image.numpy()).convert("YCbCr")
        foreground_ycbcr = np.asarray(foreground_ycbcr, dtype=np.float32)

        # Cria a cor chave em YCbCr
        # Converte o tensor para uma tupla de inteiros
        keycolor_rgb = tuple(map(int, self.key_color.numpy().flatten().tolist()))
        keycolor_pil = PIL.Image.new("RGB", (1, 1), keycolor_rgb)
        keycolor_ycbcr = np.asarray(keycolor_pil.convert("YCbCr"), dtype=np.float32).flatten()

        mask = self.create_mask(foreground_ycbcr, keycolor_ycbcr)

        # Aplica a máscara ao foreground e ao background
        out = np.zeros_like(foreground_image, dtype=np.float32)
        out = foreground_image - mask[..., None] * np.array(keycolor_rgb, dtype=np.float32)  # Usar a cor chave RGB

        if background is not None:
            out += mask[..., None] * (background.permute(1, 2, 0).numpy()).astype(np.float32)

        # Clipping para garantir que os valores estejam dentro do intervalo [0, 255]
        out = np.clip(out, 0, 255)
        out = np.uint8(out)

        return torch.tensor(out).permute(2, 0, 1)  # Retorna para [C, H, W]


class YoloBBCrop(object):
    def __init__(self, size, model="yolo11n.pt", device=None):
        self.size = size
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = YOLO(model).to(device)

    def __call__(self, clip):
        boxes = self.detection_batch(clip)

        max_box_height = 0
        for img, box in zip(clip, boxes):
            if None in box:
                continue
            x0, y0, x1, y1 = box

            frame_height, frame_width, _ = img.shape
            crop_values = self.adjust_bounding_box(x0, y0, x1, y1, frame_width, frame_height)

            boundingBoxHeight = y1 - y0
            if boundingBoxHeight > max_box_height:
                max_box_height = boundingBoxHeight
                max_box_crop_values = crop_values

        rs_crop_clip = []
        for img in clip:
            x0, y0, x1, y1 = max_box_crop_values
            cropped = img[y0:y1, x0:x1]
            rs_crop_clip.append(cv2.resize(cropped, (self.size, self.size), interpolation=cv2.INTER_LANCZOS4))
        return np.array(rs_crop_clip)

    def detection_batch(self, images):
        boxes = []

        with torch.no_grad():  # Desativa o cálculo de gradientes
            for img in images:
                # Converte a imagem em numpy array, se necessário
                if isinstance(img, PIL.Image.Image):
                    img = np.array(img)

                if isinstance(img, np.ndarray):
                    # Processa a imagem individualmente
                    results = self.model(img)

                    # Verifica e armazena as bounding boxes, se houver
                    if len(results) > 0 and len(results[0].boxes) > 0:
                        boxes.append(results[0].boxes.cpu().numpy().xyxy[0])
                    else:
                        boxes.append((None, None, None, None))
                else:
                    # Adiciona resultado vazio se a imagem não for válida
                    boxes.append((None, None, None, None))

        return boxes

    def adjust_bounding_box(self, x0, y0, x1, y1, max_width, max_height):
        """Adjusts the bounding box to ensure the correct aspect ratio."""
        bounding_box_height = y1 - y0
        bounding_box_width = x1 - x0

        if bounding_box_height < bounding_box_width:
            # Se a altura for menor que a largura, ajusta a altura
            missing_padding = bounding_box_width - bounding_box_height
            y0 -= missing_padding / 2  # Ajusta y0 para cima
            y1 += missing_padding / 2  # Ajusta y1 para baixo

            # Garante que os limites da altura estão dentro dos limites do frame
            if y0 < 0:
                y0 = 0
            if y1 > max_height:
                y1 = max_height
                y0 = y1 - bounding_box_width  # Ajusta y0 para manter a proporção
        else:
            # Garante uma proporção correta, preenchendo a largura se a altura for maior
            missing_padding = bounding_box_height - bounding_box_width
            y1 -= missing_padding / 2  # Ajusta y1 para cima
            x0 -= missing_padding / 4  # Ajusta x0 para a esquerda
            x1 += missing_padding / 4  # Ajusta x1 para a direita

            # Garante que os limites da largura estão dentro dos limites do frame
            if x0 < 0:
                x0 = 0
            if x1 > max_width:
                x1 = max_width
                x0 = x1 - bounding_box_height  # Ajusta x0 para manter a proporção

        return int(x0), int(y0), int(x1), int(y1)


class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, clip):
        clip = clip.float()
        means = torch.mean(clip, dim=(0, 2, 3), keepdim=True)
        stds = torch.std(clip, dim=(0, 2, 3), keepdim=True)

        normalized_clip = (clip - means) / (stds + 1e-8)

        return normalized_clip, means.squeeze(), stds.squeeze()


class deNormalize(object):
    def __init__(self):
        pass

    def __call__(self, clip, means, stds):
        clip = clip.float()
        return transforms.Normalize(mean=-1 * means / stds, std=1 / stds)(clip)


class HalfFps(object):
    def __init__(self):
        pass

    def __call__(self, clip):
        return clip[::2]


class ProbabilisticHalfFps(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, clip):
        if random.random() < self.prob:
            return clip[::2]
        return clip


class MotionBlurVideo:
    def __init__(self, p=0.5, kernel_size_window=[1], mode="frame_bleeding"):
        """
        Classe para aplicar motion blur ao vídeo.

        Parâmetros:
        - kernel_size: Tamanho do kernel para o motion blur.
        - mode: Modo de aplicação do blur, por padrão "frame_bleeding".
        """
        kernel_size_window = [k for k in kernel_size_window if k >= 1 or k % 2 != 0]

        self.p = p
        self.kernel_size_window = kernel_size_window
        self.kernel_size = 1
        self.mode = mode

    def __call__(self, clip: torch.Tensor):
        """
        Aplica motion blur em cada frame do vídeo.

        Parâmetros:
        - clip: Vídeo de entrada como um tensor [T, C, H, W] (T = nº de frames, C = nº de canais, H = altura, W = largura).

        Retorna:
        - blurred_clip: Vídeo com motion blur aplicado como tensor [T, C, H, W].
        """
        if random.random() > self.p:
            return clip

        self.kernel_size = random.choice(self.kernel_size_window)

        if clip.ndim != 4:
            raise ValueError("O vídeo deve ser um tensor no formato [T, C, H, W].")

        if self.mode == "frame_bleeding":
            return self.apply_frame_bleeding(clip)
        else:
            raise NotImplementedError(f"O modo '{self.mode}' não foi implementado.")

    def apply_frame_bleeding(self, clip):
        """
        Aplica blur usando o método de frame bleeding.

        Parâmetros:
        - clip: Tensor de vídeo [T, C, H, W].

        Retorna:
        - blurred_clip: Tensor de vídeo com frame bleeding aplicado [T, C, H, W].
        """

        T, C, H, W = clip.shape
        padding = self.kernel_size // 2
        padded_clip = torch.cat([clip[:padding].flip(0), clip, clip[-padding:].flip(0)], dim=0)

        blurred_clip = torch.zeros_like(clip)
        weights = torch.ones(self.kernel_size) / self.kernel_size

        for t in range(T):
            # Seleciona a janela temporal
            window = padded_clip[t : t + self.kernel_size]
            blurred_clip[t] = (weights.view(-1, 1, 1, 1) * window).sum(dim=0)  # Aplica média ponderada

        return blurred_clip


class ClipTrim:
    def __init__(self, csv_path):
        self.clip_info = self.load_csv(csv_path)

    def load_csv(self, csv_path):
        clip_info = {}
        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                video_id = row["VideoID"][:19]
                diff_start = int(row["diff_start"])
                diff_end = int(row["diff_end"])
                clip_info[video_id] = (diff_start, diff_end)
        return clip_info

    def __call__(self, clip, file_info=None):
        if file_info is None or "_signal" in file_info:
            return clip

        video_id = str(file_info)[:19]

        if video_id in self.clip_info:
            diff_start, diff_end = self.clip_info[video_id]
            clip_len = len(clip)
            if clip_len <= 1:
                return clip

            # Clamp bounds to the current clip size and force at least one frame.
            start_idx = max(0, min(int(diff_start), clip_len // 2))
            end_idx = max(1, min(clip_len, clip_len - max(0, int(diff_end))))

            # Resolve conflicting bounds (e.g., start >= end) by shrinking the start side.
            if end_idx <= start_idx:
                start_idx = max(0, end_idx - 1)

            random_start = random.randint(0, start_idx) if start_idx > 0 else 0
            min_end = max(end_idx, random_start + 1)
            random_end = random.randint(min_end, clip_len)

            clip = clip[random_start:random_end]

        return clip
