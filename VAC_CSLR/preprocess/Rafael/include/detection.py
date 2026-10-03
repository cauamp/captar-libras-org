from ultralytics import YOLO
from PIL import Image, ImageDraw
from collections import OrderedDict


def detectionBatch(frameArray: dict):
    model               = YOLO("yolo11l.pt")
    detections          = dict()
    emptyFrames         = []
    previousBoundingBox = (float('inf'), float('inf'), -1, -1)

    for idx, (frameName, frame) in enumerate(frameArray.items()):
        results = model(frame)
        for result in results:
            # Se não tiver detectado nada
            if len(result.boxes.xyxy) == 0:
                print(f"No detection for {frameName}!")
                emptyFrames.append(frameName)

                # Se a não-detecção acontecer no último frame, pega a boundingBox da última detecção
                if idx == len(frameArray) - 1:
                    for emptyFrameName in emptyFrames:
                        detections[emptyFrameName] = previousBoundingBox

                    emptyFrames = []

            # Se tiver detectado
            else:
                boundingBox = result.boxes.cpu().numpy().xyxy[0]
                detections[frameName] = boundingBox

                # Verifica se teve algum frame sem detecção antes do atual. Se tiver, 
                # interpola o tamanho da nova boundingbox levando em conta a boundingBox da última
                # detecção e a bounding box da atual
                for emptyFrameName in emptyFrames:
                    minx0 = min(boundingBox[0], previousBoundingBox[0])
                    miny0 = min(boundingBox[1], previousBoundingBox[1])
                    maxx1 = max(boundingBox[2], previousBoundingBox[2])
                    maxy1 = max(boundingBox[3], previousBoundingBox[3])

                    detections[emptyFrameName] = (minx0, miny0, maxx1, maxy1)
                
                emptyFrames         = []
                previousBoundingBox = boundingBox


    for frameName, bb in detections.items():
        frame = Image.open(frameName)
        draw  = ImageDraw.Draw(frame)
        draw.rectangle(bb, outline="red", width=5)
        draw.text((0, 0), frameName, fill="red", font_size=30)
        frame.show()

# PRECISA ser um ordered dict, que preserva a ordem de inserção. Dicionários normais
# não preservam a ordem de inserção.
frameList = OrderedDict()
for i in range(0, 4):
    frameList[f"{i}.jpg"] = Image.open(f"{i}.jpg")

detectionBatch(frameList)