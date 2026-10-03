import logging
from ultralytics import YOLO
import torch
from PIL import Image
import numpy as np

# Set the logging level to WARNING or higher to suppress info/debug logs for YOLO
logging.getLogger('ultralytics').setLevel(logging.WARNING)

''' 
    Receives a list of images (as numpy arrays or PIL.Images) and returns the
    bounding boxes for the person that's in each frame.
'''
def detection_batch(images):
    boxes = {}
    emptyFrames = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("yolo11l.pt").to(device)
    previousBoundingBox = (float('inf'), float('inf'), -1, -1)
    
    with torch.no_grad():  # Desativa o cálculo de gradientes
        for idx, img in enumerate(images):
            # Converte a imagem em numpy array, se necessário
            if isinstance(img, Image.Image):
                img = np.array(img)       
                                      
            if isinstance(img, np.ndarray):
                # Processa a imagem individualmente
                
                try:
                    result = model(img)[0]
                except Exception as e:
                    print(f"Erro ao processar frame {idx}: {str(e)}")
                    continue
                
                if len(result.boxes.xyxy) == 0:
                    print(f"Nenhuma detecção para frame {idx}!")
                    emptyFrames.append(idx)

                    # Se a não-detecção acontecer no último frame, pega a boundingBox da última detecção
                    if idx == len(images) - 1:
                        for emptyFrameIdx in emptyFrames:
                            boxes[emptyFrameIdx] = previousBoundingBox

                        emptyFrames = []

                # Se tiver detectado
                else:
                    boundingBox = tuple(result.boxes.cpu().numpy().xyxy[0])
                    boxes[idx] = boundingBox

                    # Verifica se teve algum frame sem detecção antes do atual. Se tiver, 
                    # interpola o tamanho da nova boundingbox levando em conta a boundingBox da última
                    # detecção e a bounding box da atual
                    for emptyFrameIdx in emptyFrames:
                        minx0 = min(boundingBox[0], previousBoundingBox[0])
                        miny0 = min(boundingBox[1], previousBoundingBox[1])
                        maxx1 = max(boundingBox[2], previousBoundingBox[2])
                        maxy1 = max(boundingBox[3], previousBoundingBox[3])

                        boxes[emptyFrameIdx] = (minx0, miny0, maxx1, maxy1)
                    
                    emptyFrames         = []
                    previousBoundingBox = boundingBox
            else:
                raise TypeError('At Expected list of PIL.Image or numpy arrays')
    return boxes