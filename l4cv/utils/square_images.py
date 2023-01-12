import os

import argparse

import numpy as np
import skimage
import skimage.io as sio


def square_images(directory):

    dir_list = os.listdir(directory)

    print(dir_list)

    for element in dir_list:

        image_path = os.path.join(directory, element)

        if os.path.isdir(image_path):
            square_images(image_path)

        elif image_path.endswith("jpg") or image_path.endswith("png"):
            my_image =  sio.imread(image_path)

            dim_x, dim_y = my_image.shape[0], my_image.shape[1]

            min_dim = min([dim_x, dim_y])

            crop_x = (dim_x - min_dim) // 2
            crop_y = (dim_y - min_dim) // 2

            if dim_x > dim_y and crop_x:
                my_image = my_image[crop_x:-crop_x, :, ...]
            elif crop_y:
                my_image = my_image[:, crop_y:-crop_y, ...]

            sio.imsave(image_path, my_image)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", default="data/cats_and_dogs")

    args = parser.parse_args()

    folder = args.input_folder

    square_images(folder)
