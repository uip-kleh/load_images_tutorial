import numpy as np
from PIL import Image

def load_image(fname: str) -> np.array:
    image = Image.open(fname)
    # image.show()

    return np.array(image)
    
    
if __name__ == "__main__":
    fname = "images/classA/0.png"
    image = load_image(fname)
    print(image.shape)