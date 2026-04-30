from PIL import Image, ImageOps
import os
import numpy as np


_IMAGES_FOLDER = "./sample_images/"

def image_dict(imgPath):
    #open image and get the info
    img_size_kb = os.stat(imgPath).st_size/1024
    img_open = Image.open(imgPath)
    data = img_open.getdata()

    print(img_open.size[0])
    print(img_open.size[1])

    #Attempts to reshape the flat pixel data into a 3D array with dimensions (width, height, channels). If the image is grayscale, the -1 will resolve to 1 (one channel for intensity).If the image is RGB, the -1 will resolve to 3 (three channels for Red, Green, Blue).
    img_pixels = np.array(data).reshape(*img_open.size, -1) 
    print(img_pixels)



def main():
    #Get names of images in folder sample_images
    photos_list = os.listdir(_IMAGES_FOLDER)

    photos_list_path = []
    for photo in photos_list:
        photos_list_path.append(_IMAGES_FOLDER + photo)

    image_dictionary = []
    for photo in photos_list_path:
        image_dictionary.append(image_dict(photo))
    
    
        





if __name__ == "__main__":
    main()
