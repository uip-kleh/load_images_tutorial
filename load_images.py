import os
import glob
from load_image import load_image

def load_images(dname: str) -> list:
    images = []
    list = glob.glob(dname)
    for fname in list:
        image = load_image(fname)
        images.append(image)
    return images

    
if __name__ == "__main__":
    dname = "images/classA/*"
    images = load_images(dname)
    print(images)