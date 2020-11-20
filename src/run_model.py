from unet import UNet
import os, glob, torch, tqdm
from os.path import join
from preprocess import tensorize_image, decode_and_convert_image
from mask_on_image import write_mask_on_image2
import numpy as np 

# PARAMETERS
cuda = True
test_size  = 0.1
model_file_name = "test3"
predict_save_file_name = "test3" + "_run_model"
input_shape = (224, 224)
n_classes = 2

SRC_DIR = os.getcwd()
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
# MASK_DIR = os.path.join(DATA_DIR, 'masks')
# IMAGE_DIR = os.path.join(DATA_DIR, 'test_images')
# MASK_DIR = os.path.join(DATA_DIR, 'test_masks')
MODEL_DIR = os.path.join(DATA_DIR, 'models')

# PREPARE IMAGE AND MASK LISTS
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()

# SHUFFLE INDICES
indices = np.random.permutation(len(image_path_list))

# DEFINE TEST AND VALID INDICES
test_ind  = int(len(indices) * test_size)

# SLICE TEST DATASET FROM THE WHOLE DATASET
test_input_path_list = image_path_list[:test_ind]


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
        mask = decoded_list[0]
        predict_mask_list.append(mask)

        write_mask_on_image2(predict_mask_list, images, input_shape, predict_save_file_name)


if __name__ == "__main__":
    loaded_model = load_model(join(MODEL_DIR, model_file_name + ".pt"))
    test(loaded_model, test_input_path_list)