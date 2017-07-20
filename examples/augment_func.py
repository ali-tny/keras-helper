import random
from PIL import Image

def augment_func(img):
    """Example augmentation function to be given to ImageDataGenerator. Takes a 
    PIL.Image object and randomly augments it in a few ways."""

    rotation = random.choice([0,1,2,3])
    extra_rotation = random.choice(range(6))
    h_reflect = random.choice([0,1])
    v_reflect = random.choice([0,1])

    if rotation == 1:
        img = img.transpose(Image.ROTATE_90)
    if rotation == 2:
        img = img.transpose(Image.ROTATE_180)
    if rotation == 3:
        img = img.transpose(Image.ROTATE_270)

    img.rotate(extra_rotation)

    if h_reflect:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if v_reflect:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    return img
