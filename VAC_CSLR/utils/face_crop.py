from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from ultralytics import YOLO
import torch
import cv2
import numpy as np

class YoloFaceCrop(object):
    """
    Classe para recortar rostos em clipes de vídeo usando o modelo YOLO (You Only Look Once).

    Parâmetros:
    -----------
    size : int
        O tamanho do recorte em pixels. A imagem será redimensionada para ter o maior lado igual a `size` antes do recorte.
        
    ratio : float, opcional (default=0.1)
        Razão que define a frequência em que o centro do rosto é recalculado. Por exemplo, se `ratio=0.1`,
        o centro será recalculado a cada 10% dos frames.
        
    max_workers : int, opcional (default=None)
        Número máximo de threads usadas para processamento em paralelo. Se `None`, utiliza o número de CPUs disponíveis.
    """
    def __init__(self, size, ratio = 0.01, max_workers = None):
        self.size = size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = YOLO("yolov5s.pt").to(self.device)

        self.ratio = ratio
        if not max_workers:
            self.max_workers = os.cpu_count()

    def __call__(self, clip):
        if not isinstance(clip[0], np.ndarray):
            raise TypeError(f'At {self.__class__} Expected list of numpy.ndarray' +
                            'but got list of {0}'.format(type(clip[0])))
        cropped_clip = []
        crop_size = self.size

        def calculate_center(frame):
            # Garantir que os valores estejam na faixa [0, 255] e o tipo de dados seja uint8
            frame_np = np.clip(frame, 0, 255).astype(np.uint8)

            h, w, _ = frame_np.shape
            aspect_ratio = w / h
            
            if w < h:
                new_width = self.size
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = self.size
                new_width = int(new_height * aspect_ratio)

            frame_resized_np = cv2.resize(frame_np, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Fazer a detecção de objetos no frame com YOLO
            results = self.yolo_model(frame_resized_np)

            # Filtrar apenas detecções de rostos (classe 'person' ou 'face', dependendo da versão do YOLO)
            # YOLOv5 detecta 'person', portanto, filtramos por classe 0 (person)
            detections = results[0].boxes  # Predições no primeiro item (primeiro frame)
            if len(results) == 0 or len(results[0].boxes) == 0:
                print('Nenhum rosto foi detectado.')
                return None, None
            face_boxes = detections.xyxy  # Filtra classe 0 (person)
            
            # Obter a primeira detecção (a maior probabilidade, ou seja, o primeiro bounding box)
            x_min, y_min, x_max, y_max = face_boxes[0].int().tolist()

            # Calcular o centro do rosto
            x = int((x_min + x_max) / 2)
            y = int((y_min + y_max) / 2)
                
            return x, y
        
        
        def process_frame(i, frame, centers):      
            # Garantir que os valores estejam na faixa [0, 255] e o tipo de dados seja uint8
            frame_np = np.clip(frame, 0, 255).astype(np.uint8)

            h, w, _ = frame_np.shape
            aspect_ratio = w / h
            
            if w < h:
                new_width = self.size
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = self.size
                new_width = int(new_height * aspect_ratio)

            frame_resized_np = cv2.resize(frame_np, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
                
            # Tamanho do crop
            # Calcular as coordenadas do crop
            crop_x_min = max(0, centers['x'] - crop_size // 2)
            crop_y_min = max(0, centers['y'] - crop_size // 2)
            crop_x_max = min(frame_resized_np.shape[1], crop_x_min + crop_size)
            crop_y_max = min(frame_resized_np.shape[0], crop_y_min + crop_size)
                
            # Ajustar caso o crop ultrapasse os limites da imagem
            if crop_x_max - crop_x_min < crop_size:
                crop_x_min = max(0, crop_x_max - crop_size)
            if crop_y_max - crop_y_min < crop_size:
                crop_y_min = max(0, crop_y_max - crop_size)
            
            # Realizar o crop usando slicing do NumPy
            cropped_image = frame_resized_np[crop_y_min:crop_y_max, crop_x_min:crop_x_max]           
        
            return i, cropped_image

        centers_x = []
        centers_y = []
        
        frame_indices = range(0, len(clip), max(1, int(self.ratio * len(clip))))

        for i in frame_indices:
            x, y = calculate_center(clip[i])
            if x is not None and y is not None:
                centers_x.append(x)
                centers_y.append(y)
        
        mean_center_x = int(np.mean(centers_x)) if centers_x else 0
        mean_center_y = int(np.mean(centers_y)) if centers_y else 0
        centers = {'x': mean_center_x, 'y': mean_center_y}
        
        if centers_x and centers_y:
            print(f'Diferença no eixo X entre o centro médio e o centro do primeiro frame: {mean_center_x - centers_x[0]} ({(mean_center_x - centers_x[0])/self.size*100:.2f}%)')
            print(f'Diferença no eixo Y entre o centro médio e o centro do primeiro frame: {mean_center_y - centers_y[0]} ({(mean_center_y - centers_y[0])/self.size*100:.2f}%)')
        else:
            print('Nenhum rosto foi detectado para calcular as diferenças nos centros.')
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_frame, i, frame, centers) for i, frame in enumerate(clip)]
            for future in as_completed(futures):
                i, cropped_image = future.result()
                if cropped_image is not None:
                    cropped_clip.append((i, cropped_image))
                    
        # Ordenar para garantir que os frames estejam na ordem correta
        cropped_clip.sort(key=lambda x: x[0])
        
        # Extrair apenas os tensores (sem os índices) e empilhar
        cropped_clip = [img for _, img in cropped_clip]
        return cropped_clip