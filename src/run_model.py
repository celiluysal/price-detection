from unet import UNet
import os, glob, torch, tqdm, cv2
from os.path import join
from preprocess import tensorize_image, decode_and_convert_image
# from mask_on_image import write_mask_on_image2, crop_prize, write_mask_bbox_on_image
import numpy as np 
from PIL import Image

# PARAMETERS
cuda = True
test_size  = 0.1
model_file_name = "test2"
predict_save_file_name = model_file_name + "_predict"
cropped_save_file_name = model_file_name + "_cropped"

input_shape = (224, 224)
n_classes = 2

SRC_DIR = os.getcwd()
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(DATA_DIR, 'models')
IMAGE_DIR = os.path.join(DATA_DIR, 'model_input_images')

MODEL_OUT = os.path.join(DATA_DIR, 'model_outputs')
if not os.path.exists(MODEL_OUT):
    os.mkdir(MODEL_OUT)

image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()



def load_model(model_dir):
    loaded_model = torch.load(model_dir)
    loaded_model.eval()
    loaded_model = loaded_model.cuda()
    return loaded_model

def test(model, images):
    predict_mask_list = list()
    for image in tqdm.tqdm(images):
        batch_input = tensorize_image([image], input_shape, cuda)
        output = model(batch_input)

        label = output > 0.5

        decoded_list = decode_and_convert_image(label, n_class=2)
        mask = decoded_list[0].astype(np.uint8)

        predict_mask_list.append(mask)

    write_mask_on_image(predict_mask_list, images, (1024,768))
    crop_price(predict_mask_list, images, (1024,768))


def write_mask_on_image(mask_list, image_file_names, shape):
    save_file_name = os.path.join(MODEL_OUT, predict_save_file_name)
    if not os.path.exists(save_file_name):
        os.mkdir(save_file_name)

    for mask, image_file_name in zip(mask_list, image_file_names):

        image = cv2.imread(image_file_name).astype(np.uint8)
        image = cv2.resize(image, shape)
        mask = cv2.resize(mask, shape)
        x,y,w,h = cv2.boundingRect(mask)

        selected_mask = find_biggest_area(mask)

        mask_image = image.copy()
        mask_ind = mask == 255
        mask_image[mask_ind, :] = (102, 204, 102)
        opac_image = (image/2 + mask_image/2).astype(np.uint8)

        cv2.rectangle(opac_image, (x, y), (x + w, y + h), (0,255,255), 4)
        x,y,w,h = cv2.boundingRect(selected_mask)
        cv2.rectangle(opac_image, (x, y), (x + w, y + h), (0,0,255), 4)

        image_name = image_file_name.split('/')[-1].split('.')[0]
        cv2.imwrite(join(save_file_name, image_name+ "_predict" + ".png"), opac_image)

def find_biggest_area(mask):
    thresh = mask.copy()
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    area_list = list()
    for c in cnts:
        area_list.append(cv2.contourArea(c))
        
    if (len(area_list)):
        max_value = max(area_list)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < max_value:
                cv2.drawContours(thresh, [c], -1, (0,255,0), -1)
    return thresh

def crop_price(mask_list, image_file_names, shape):
    save_file_name = os.path.join(MODEL_OUT, cropped_save_file_name)
    if not os.path.exists(save_file_name):
        os.mkdir(save_file_name)

    for mask, image_file_name in zip(mask_list, image_file_names):

        image = cv2.imread(image_file_name).astype(np.uint8)
        image = cv2.resize(image, shape)
        mask = cv2.resize(mask, shape)

        selected_mask = find_biggest_area(mask)
        x,y,w,h = cv2.boundingRect(selected_mask)
        crop_image = image[y:y+h, x:x+w]

        image_name = image_file_name.split('/')[-1].split('.')[0]

        if(w or h):
            cv2.imwrite(join(save_file_name, image_name + "_cropped" + ".png"), crop_image)
        else:
            print(image_name, " is empty" , x,y,w,h)


if __name__ == "__main__":
    loaded_model = load_model(join(MODEL_DIR, model_file_name + ".pt"))
    test(loaded_model, image_path_list)