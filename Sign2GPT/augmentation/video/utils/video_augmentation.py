# ----------------------------------------
# Written by Yuecong Min
# ----------------------------------------
import cv2
import PIL
import torch
import random
import numbers
import numpy as np
from ultralytics import YOLO
import logging

logging.getLogger('ultralytics').setLevel(logging.WARNING)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class KorniaCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label, file_info=None):
        info = {}
        for t in self.transforms:
            if isinstance(t, Normalize):
                image, means, std = t(image)
                info['means'] = means
                info['stds'] = std
            else:
                image = t(image)
        return image, info


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
                raise ValueError(
                    "O numpy.ndarray deve ter 4 dimensões: (N, altura, largura, canais)")
        return video_list


class ToNumpyArray(object):
    def __call__(self, video):
        if isinstance(video, torch.Tensor):
            video = video.permute(0, 2, 3, 1).cpu().numpy()
        return video


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
        top = int(round((im_h - new_h) / 2.))
        left = int(round((im_w - new_w) / 2.))
        return [img[top:top + new_h, left:left + new_w] for img in clip]


class YoloBBCrop(object):
    def __init__(self, size, model="yolo11n.pt", device=None):
        self.size = size
        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = YOLO(model).to(device)

    def __call__(self, clip):
        boxes = self.detection_batch(clip)

        max_box_height = 0
        for img, box in zip(clip, boxes):
            if None in box:
                continue
            x0, y0, x1, y1 = box

            frame_height, frame_width, _ = img.shape
            crop_values = self.adjust_bounding_box(
                x0, y0, x1, y1, frame_width, frame_height)

            boundingBoxHeight = y1 - y0
            if boundingBoxHeight > max_box_height:
                max_box_height = boundingBoxHeight
                max_box_crop_values = crop_values

        rs_crop_clip = []
        for img in clip:
            x0, y0, x1, y1 = max_box_crop_values
            cropped = img[y0:y1, x0:x1]
            rs_crop_clip.append(cv2.resize(
                cropped, (self.size, self.size), interpolation=cv2.INTER_LANCZOS4))
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
