import numpy as np

# https://www.youtube.com/watch?v=WuVyG4pg9xQ&t=138s
def equalizeImg(image):
    # Convert the image to float32 for improved precision during calculations
    imageFloat   = image.astype(np.float32)

    # Find the maximum pixel value in the image
    maxPixel = 160

    # Scale the pixel values to push the brightest pixel to 255
    # Calculate the scaling factor
    scaleFactor = 255 / maxPixel

    # Expand the histogram by scaling pixel values
    equalizedImage = imageFloat * scaleFactor

    # Clip the values to ensure they are in the range [0, 255]
    equalizedImage = np.clip(equalizedImage, 0, 255)

    # Convert back to uint8
    equalizedImage = equalizedImage.astype(np.uint8)

    return equalizedImage
