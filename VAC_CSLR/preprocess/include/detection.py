import logging
from ultralytics import YOLO
import torch

# Set the logging level to WARNING or higher to suppress info/debug logs
logging.getLogger('ultralytics').setLevel(logging.WARNING)

''' 
    Receives a numpy representation of the image and returns the
    bounding boxes for the person that's in the frame.
'''
def detection(img):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("yolo11x.pt").to(device)

    with torch.no_grad():  # Desativa o cálculo de gradientes
        results = model(img)
    
    if len(results) == 0 or len(results[0].boxes) == 0:
        return (None, None, None, None)
    
    return results[0].boxes.cpu().numpy().xyxy[0]