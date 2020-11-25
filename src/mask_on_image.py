import os, cv2, tqdm
import numpy as np
from os import listdir
from os.path import isfile, join

MASK_DIR  = '../data/masks'
IMAGE_DIR = '../data/images'
IMAGE_OUT_DIR = '../data/masked_images'

if not os.path.exists(IMAGE_OUT_DIR):
    os.mkdir(IMAGE_OUT_DIR)

def image_mask_check(image_path_list, mask_path_list):
    for image_path, mask_path in zip(image_path_list, mask_path_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        mask_name  = mask_path.split('/')[-1].split('.')[0]
        assert image_name == mask_name, "Image and mask name does not match {} - {}".format(image_name, mask_name)

def write_mask_on_image():
    image_file_names = [f for f in listdir(IMAGE_DIR) if isfile(join(IMAGE_DIR, f))]
    mask_file_names = [f for f in listdir(MASK_DIR) if isfile(join(MASK_DIR, f))]

    image_file_names.sort()
    mask_file_names.sort()

    image_mask_check(image_file_names,mask_file_names)

    for image_file_name, mask_file_name in tqdm.tqdm(zip(image_file_names, mask_file_names)):
        
        image_path = os.path.join(IMAGE_DIR, image_file_name)
        mask_path = os.path.join(MASK_DIR, mask_file_name)
        mask  = cv2.imread(mask_path, 0).astype(np.uint8)
        image = cv2.imread(image_path).astype(np.uint8)

        mask_image = image.copy()
        mask_ind = mask == 255
        mask_image[mask_ind, :] = (102, 204, 102)
        opac_image = (image/2 + mask_image/2).astype(np.uint8)
        
        cv2.imwrite(join(IMAGE_OUT_DIR, mask_file_name), opac_image)

if __name__ == '__main__':
    write_mask_on_image()