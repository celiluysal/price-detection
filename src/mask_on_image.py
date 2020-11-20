
import os, cv2, tqdm
import numpy as np
from os import listdir
from os.path import isfile, join

MASK_DIR  = '../data/masks'
IMAGE_DIR = '../data/images'

# MASK_DIR  = '../data/test_masks'
# IMAGE_DIR = '../data/test_images'
IMAGE_OUT_DIR = '../data/test_masked_images2'
IMAGE_OUT_DIR2 = '../data/predict3_images'
# IMAGE_OUT_DIR2 = '../data/predict_images'
# IMAGE_OUT_DIR = '../data/test_masked_images_224x224'


if not os.path.exists(IMAGE_OUT_DIR):
    os.mkdir(IMAGE_OUT_DIR)
    
if not os.path.exists(IMAGE_OUT_DIR2):
    os.mkdir(IMAGE_OUT_DIR2)


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

        # output_shape = (224,224)
        # mask = cv2.resize(mask,output_shape)
        # image = cv2.resize(image,output_shape)

        mask_image = image.copy()
        mask_ind = mask == 255
        mask_image[mask_ind, :] = (0, 0, 255)
        
        cv2.imwrite(join(IMAGE_OUT_DIR, mask_file_name), mask_image)
        if False:
            cv2.imshow('o', image)
            cv2.waitKey(1)

def write_mask_on_image2(mask_list, image_file_names, shape, save_file_name):
    save_file_name = "../data/predicts/" + save_file_name
    if not os.path.exists(save_file_name):
        os.mkdir(save_file_name)

    for mask, image_file_name in zip(mask_list, image_file_names):

        image_path = os.path.join(IMAGE_DIR, image_file_name)
        image = cv2.imread(image_path).astype(np.uint8)
        image = cv2.resize(image, shape)
        
        mask_image = image.copy()
        mask_ind = mask == 255
        mask_image[mask_ind, :] = (0, 0, 255)

        image_name = image_path.split('/')[-1].split('.')[0]
        cv2.imwrite(join(save_file_name, image_name+".png"), mask_image)


if __name__ == '__main__':
    write_mask_on_image()