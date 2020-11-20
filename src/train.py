from unet import UNet
from mask_on_image import write_mask_on_image2
from preprocess import tensorize_image,tensorize_mask, image_mask_check, decode_and_convert_image
from data_utils import draw_loss_graph, norm

from os.path import join
import os, glob, tqdm, cv2, torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


######### PARAMETERS ##########
valid_size = 0.3
test_size  = 0.1
batch_size = 8
epochs = 20
cuda = True
augmentation = False
test_predict = True
predict_save_file_name = "test3"
model_file_name = predict_save_file_name
input_shape = (224, 224)
n_classes = 2
###############################

######### DIRECTORIES #########
SRC_DIR = os.getcwd()
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')
# IMAGE_DIR = os.path.join(DATA_DIR, 'test_images')
# MASK_DIR = os.path.join(DATA_DIR, 'test_masks')
MODEL_DIR = os.path.join(DATA_DIR, 'models')
##############################

# PREPARE IMAGE AND MASK LISTS
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()

mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()

# DATA CHECK
image_mask_check(image_path_list, mask_path_list)

# SHUFFLE INDICES
indices = np.random.permutation(len(image_path_list))

# DEFINE TEST AND VALID INDICES
test_ind  = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)

# SLICE TEST DATASET FROM THE WHOLE DATASET
test_input_path_list = image_path_list[:test_ind]
test_label_path_list = mask_path_list[:test_ind]

# SLICE VALID DATASET FROM THE WHOLE DATASET
valid_input_path_list = image_path_list[test_ind:valid_ind]
valid_label_path_list = mask_path_list[test_ind:valid_ind]

# SLICE TRAIN DATASET FROM THE WHOLE DATASET
train_input_path_list = image_path_list[valid_ind:]
train_label_path_list = mask_path_list[valid_ind:]

# DEFINE STEPS PER EPOCH
steps_per_epoch = len(train_input_path_list)//batch_size

# CALL MODEL
model = UNet(n_channels=3, n_classes=2, bilinear=True)

# DEFINE LOSS FUNCTION AND OPTIMIZER
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.RMSprop(model.parameters(),lr=0.002, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.002)

# IF CUDA IS USED, IMPORT THE MODEL INTO CUDA
if cuda:
    model = model.cuda()

# TRAINING THE NEURAL NETWORK
run_loss_list = list()
val_loss_list = list()

def train():
    for epoch in range(epochs):
        running_loss = 0
        for ind in tqdm.tqdm(range(steps_per_epoch)):
            batch_input_path_list = train_input_path_list[batch_size*ind:batch_size*(ind+1)]
            batch_label_path_list = train_label_path_list[batch_size*ind:batch_size*(ind+1)]
            
            # batch_input = tensorize_image(batch_input_path_list, input_shape, cuda, epoch%2==0 if augmentation else False)
            batch_input = tensorize_image(batch_input_path_list, input_shape, cuda, augmentation)
            batch_label = tensorize_mask(batch_label_path_list, input_shape, n_classes, cuda)

            optimizer.zero_grad()

            outputs = model(batch_input)

            loss = criterion(outputs, batch_label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if ind == steps_per_epoch-1:
                run_loss_list.append(running_loss)
                str1 = 'training loss on epoch   {}: {}'.format(epoch, running_loss)
                val_loss = 0
                for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
                    batch_input = tensorize_image([valid_input_path], input_shape, cuda)
                    batch_label = tensorize_mask([valid_label_path], input_shape, n_classes, cuda)
                    
                    outputs = model(batch_input)

                    loss = criterion(outputs, batch_label)
                    loss.backward()
                    val_loss += loss.item()

                val_loss_list.append(val_loss)
                str2 = 'validation loss on epoch {}: {}'.format(epoch, val_loss)
        print(str1)
        print(str2)

    norm_run_loss_list = norm(run_loss_list)
    norm_val_loss_list = norm(val_loss_list)

    graph_name = predict_save_file_name
    draw_loss_graph(epochs, norm_run_loss_list, norm_val_loss_list, graph_name)

def test(_model:UNet):    
    if test_predict:
        torch.cuda.empty_cache()
        predict_mask_list = []
        for test_input_path, test_label_path in tqdm.tqdm(zip(test_input_path_list, test_label_path_list)):
            batch_input = tensorize_image([test_input_path], input_shape, cuda)
            # batch_label = tensorize_mask([test_label_path], input_shape, n_classes, cuda)
                    
            outputs = _model(batch_input)

            label = outputs > 0.5
            decoded_list = decode_and_convert_image(label, n_class=2)
            mask = decoded_list[0]
            predict_mask_list.append(mask)
                    
            write_mask_on_image2(predict_mask_list, test_input_path_list, input_shape, predict_save_file_name)

def save_model(model, model_name):
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    model_name += ".pt"
    torch.save(model,join(MODEL_DIR, model_name))

def load_model(model_dir):
    loaded_model = torch.load(model_dir)
    loaded_model.eval()
    return loaded_model

if __name__ == "__main__":
    train()
    save_model(model, model_file_name)
    test(model)
    
    print("test is done. Press any key for load and test model")
    input()

    predict_save_file_name += "with_load"

    loaded_model = load_model(join(MODEL_DIR, model_file_name + ".pt"))
    test(loaded_model)