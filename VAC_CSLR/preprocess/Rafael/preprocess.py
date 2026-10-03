# Para o test, dev e train em dataset_location, abre cada um e acha as imagens em cada um.
# Para cada imagem, preprocessa. A ordem é: Color balance -> Remoção de ruído -> Crop -> Resize para 255x255 -> Salvar a imagem


from include.histogram_equalization import equalizeImg
from include.parser import make_parser, validateParams
from include.detection import detection

from PIL import Image, ImageDraw

import numpy as np
import os


def main():
    args = make_parser().parse_args()
    validateParams(args)
    
    # For each subdivision of the dataset (train, test, dev)
    for subset in ["train", "test", "dev"]:
        subsetPath = os.path.join(args.dataset_location, subset)

        if not os.path.isdir(subsetPath):
            print(f"Warning: {subsetPath} does not exist.")
            continue

        # For each subfolder holding the extracted frames of each video
        for subfolder in os.listdir(subsetPath):
            
            subfolderPath = os.path.join(subsetPath, subfolder)

            if not os.path.isdir(subfolderPath):
                print(f"Warning: {subfolderPath} is not a directory.")
                continue

            # For each image contained in the subfolder
            for imgName in os.listdir(subfolderPath):
                imgPath = os.path.join(subfolderPath, imgName)


                if not imgName.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    print(f"Warning: {imgPath} is not a valid image file.")
                    continue
                
                # Open the image
                img = Image.open(imgPath)
                img = np.asarray(img)

                # Equalize the image to brighten it up
                img = equalizeImg(img)
                img = Image.fromarray(img)

                # Find the bounding box for the person in the frame
                x0, y0, x1, y1 = detection(img)
                
                boundingBoxHeight = y1 - y0
                boundingBoxWidth  = x1 - x0
                
                if boundingBoxHeight < boundingBoxWidth:
                    raise ValueError(f"boundingBoxHeight cannot be smaller than boundingBoxWidth. Please, check the source image!\nSource img name: {imgName}")
                # missingPadding calculates how many pixels are necessary to make the bounding box
                # have the aspect ratio 1:1. In other words "How many pixels wider does the box have to be?"
                missingPadding = (boundingBoxHeight - boundingBoxWidth)

                # Instead of just growing the box to the sides, I decided to also remove some pixels from the bottom. This makes the box
                # "zoom in" a little bit to the center, and gives the image a greater resolution to capture more details in the face and hands.

                # Removes missingPadding/2 pixels from the bottom
                y1 = y1 - ( missingPadding / 2 )

                # Grows the left side of the bounding box by missingPadding/4 pixels
                x0 = x0 - ( missingPadding / 4 )
                # Grows the right side of the bounding box
                x1 = x1 + ( missingPadding / 4 )


                cropped = img.crop((x0, y0, x1, y1))
                resized = cropped.resize((255,255), resample=Image.Resampling.LANCZOS)
                
                # Saves the final result
                resized.save(f"{args.save_location}/{subset}/{imgName.split(".")[0]}_equalized.png")
                print(f"Saved {args.save_location}/{subset}/{imgName.split(".")[0]}_equalized.png")


if __name__ == '__main__':
    main()