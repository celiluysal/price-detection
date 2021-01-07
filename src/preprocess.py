import numpy as np
import cv2
import torch
import glob
from torchvision import transforms as T
from PIL import Image
from PIL import ImageOps

MASK_DIR = "..\\data\\masks"
IMG_DIR = "..\\data\\images"
# MASK_DIR = "../data/test_masks"
# IMG_DIR = "../data/test_images"

def tensorize_image(image_path, output_shape, cuda=False, augment=False):
    dataset = list()
    Transform = list()
    Transform.append(T.Resize(output_shape))
    if augment:
        Transform.append(T.ColorJitter(brightness=0.4, contrast=0.4, hue=0.06))
    Transform.append(T.ToTensor())
    Transform = T.Compose(Transform)

    for file_name in image_path:
        image = Image.open(file_name)
        image = ImageOps.exif_transpose(image)
        image = Transform(image)

        dataset.append(image)

    tensor = torch.stack(dataset)

    if cuda:
        tensor = tensor.cuda()
    return tensor
    

def tensorize_mask(mask_path, output_shape ,n_class, cuda=False):
    batch_masks = list()

    for file_name in mask_path:
        mask = cv2.imread(file_name, 0)
        mask = cv2.resize(mask, output_shape)
        # mask = mask / 255
        encoded_mask = one_hot_encoder(mask, n_class)  
        torchlike_mask = torchlike_data(encoded_mask) #[C,W,H]

        batch_masks.append(torchlike_mask)      
  
    batch_masks = np.array(batch_masks, dtype=np.int)
    torch_mask = torch.from_numpy(batch_masks).float()

    if cuda:
        torch_mask = torch_mask.cuda()
    return torch_mask

def one_hot_encode(data, n_class):
    encoded_data = np.zeros((data.shape[0], data.shape[1], n_class), dtype=np.int)
    encoded_labels = [[1,0],[0,1]]
    # for lbl in range(n_class):
    #     encoded_label = [0] * n_class 
    #     encoded_label[lbl] = 1
    #     encoded_labels.append(encoded_label)
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if ((data[i][j] == 0).all()):
                encoded_data[i, j] = encoded_labels[0]
            else: #((data[i][j] == 1).all()):
                encoded_data[i, j] = encoded_labels[1]
    return encoded_data

def one_hot_encoder(data, n_class):
    encoded_data = np.zeros((*data.shape, n_class), dtype=np.int) # (width, height, number_of_class)'lık bir array tanımlıyorum. 

    encoded_labels = [[0,1], [1,0]]
    for lbl in range(n_class):

        encoded_label = encoded_labels[lbl] # lbl = 0 için (arkaplan) [1, 0] labelini oluşturuyorum, 
                                # lbl = 1 için (freespace) [0, 1] labelini oluşturuyorum.
        numerical_class_inds = data[:,:] == lbl # lbl = 0 için data'nın 0'a eşit olduğu w,h ikililerini alıyorum diyelim ki (F).
                                                # lbl = 1 için data'nın 1'e eşit olduğu w,h ikililerini alıyorum diyelim ki (O).
        encoded_data[numerical_class_inds] = encoded_label # lbl = 0 için tüm F'in sahip olduğu tüm w,h ikililerini [1, 0]'a eşitliyorum.
                                                            # lbl = 1 için tüm O'un sahip olduğu tüm w,h ikililerini [0, 1]'e eşitliyorum.
    return encoded_data

def decode_and_convert_image(data, n_class):
    decoded_data_list = []
    decoded_data = np.zeros((data.shape[2], data.shape[3]), dtype=np.int)

    for tensor in data:
        for i in range(len(tensor[0])):
            for j in range(len(tensor[1])):
                if (tensor[1][i,j] == 0):
                    decoded_data[i, j] = 255
                else: #(tensor[1][i,j] == 1):
                    decoded_data[i, j] = 0
        # print(decoded_data)
        
        # image.show()
        # plt.imshow(decoded_data, cmap="gray")
        # plt.show()
        decoded_data_list.append(decoded_data)
    
    # decoded_data_list[0].show() 

    return decoded_data_list
    


def torchlike_data(data):
    n_channels = data.shape[2]
    torchlike_data = np.empty((n_channels, data.shape[0], data.shape[1]))
    for ch in range(n_channels):
        torchlike_data[ch] = data[:,:,ch]
    # print((torchlike_data).shape)
    return torchlike_data

def image_mask_check(image_path_list, mask_path_list):
    for image_path, mask_path in zip(image_path_list, mask_path_list):
        image_name = image_path.split('\\')[-1].split('.')[0]
        mask_name  = mask_path.split('\\')[-1].split('.')[0]
        assert image_name == mask_name, "Image and mask name does not match {} - {}".format(image_name, mask_name)

if __name__ == '__main__':
    
    image_file_names = glob.glob(IMG_DIR + "\\*")
    image_file_names.sort()
    batch_image_list = image_file_names[:1] #first n
    batch_image_tensor = tensorize_image(batch_image_list, (224,224), augment=False)
    # print(batch_image_tensor[0])
    
    print(batch_image_tensor.dtype)
    print(type(batch_image_tensor))
    print(batch_image_tensor.shape)

    # print(len(batch_image_tensor))

    # plt.figure(figsize=(4,4)) # specifying the overall grid size

    # for i in range(16):
    #     plt.subplot(4,4,i+1)    # the number of images in the grid is 5*5 (25)
    #     plt.imshow(batch_image_tensor[i].permute(1,2,0))
    # plt.show()

    # for i in range(35):
    #     print("i:",i," - ", 0==i%3)


    print("------------")    
    
    mask_file_names = glob.glob(MASK_DIR + "\\*")
    mask_file_names.sort()
    batch_mask_list = mask_file_names[:1] #first n
    batch_mask_tensor = tensorize_mask(batch_mask_list, (224,224), 2)
    
    print(batch_mask_tensor.dtype)
    print(type(batch_mask_tensor))
    print(batch_mask_tensor.shape)  

    # print(batch_mask_tensor[0])

    # image_list = decode_and_convert_image(batch_mask_tensor,2)
    # img = image_list[0]
    # print(type(img))
    # print(img.shape)
    # plt.imshow(img, cmap="gray")
    # plt.show()