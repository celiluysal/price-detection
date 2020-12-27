from unet import UNet
import torch, cv2, copy
import numpy as np 
from torchvision import transforms as T
from PIL import Image

# PARAMETERS
cuda = False
input_shape = (224, 224)
n_classes = 2

# DIRECTORIES
model_file_name = "test8"
DATA_DIR = "..\\data"
MODEL_DIR = DATA_DIR + "\\models\\" + model_file_name + ".pt"


def load_model(model_dir):
    model = UNet(n_channels=3, n_classes=2, bilinear=True)

    if cuda:
        model = model.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model.load_state_dict(copy.deepcopy(torch.load(model_dir,device)))
    return model

    
def make_prediction(image):
    model = load_model(MODEL_DIR)    
    
    batch_input = tensorize_image(image, input_shape, cuda)
    output = model(batch_input)
    label = output > 0.5

    decoded_mask = decode_and_convert_image(label, n_class=2)
    predict_mask = decoded_mask.astype(np.uint8)
    
    cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # masked_image = mask_on_image(predict_mask, cv2_image)
    
    # show_image = cv2.resize(masked_image,(1024,768))
    # cv2.imshow("dvf",show_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    cropped_image = crop_price(predict_mask, cv2_image)
    
    # show_image = cv2.resize(cropped_image,(1024,768))
    # cv2.imshow("dvf",cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return cropped_image
    

def tensorize_image(image, output_shape, cuda=False):
    dataset = list()
    Transform = list()
    Transform.append(T.Resize(output_shape))
    Transform.append(T.ToTensor())
    Transform = T.Compose(Transform)
    
    image = Transform(image)
    dataset.append(image)
    tensor = torch.stack(dataset)
    
    if cuda:
        tensor = tensor.cuda()
    return tensor

def decode_and_convert_image(data, n_class):
    decoded_data = np.zeros((data.shape[2], data.shape[3]), dtype=np.int)

    for tensor in data:
        for i in range(len(tensor[0])):
            for j in range(len(tensor[1])):
                if (tensor[1][i,j] == 0):
                    decoded_data[i, j] = 255
                else: #(tensor[1][i,j] == 1):
                    decoded_data[i, j] = 0

    return decoded_data

def crop_price(mask, image):
    mask = cv2.resize(mask, (image.shape[1],image.shape[0]))

    selected_mask = find_biggest_area(mask)
    x,y,w,h = cv2.boundingRect(selected_mask)
    crop_image = image[y:y+h, x:x+w]

    if(w or h):
        return crop_image
    else:
        print("image is empty" , x,y,w,h)
        return 


def mask_on_image(mask, image):
    mask = cv2.resize(mask, (image.shape[1],image.shape[0]))
    x,y,w,h = cv2.boundingRect(mask)

    selected_mask = find_biggest_area(mask)

    mask_image = image.copy()
    mask_ind = mask >= 125
    mask_image[mask_ind, :] = (102, 204, 102)
    opac_image = (image/2 + mask_image/2).astype(np.uint8)

    cv2.rectangle(opac_image, (x, y), (x + w, y + h), (0,255,255), 4)
    x,y,w,h = cv2.boundingRect(selected_mask)
    cv2.rectangle(opac_image, (x, y), (x + w, y + h), (0,0,255), 4)
    
    return opac_image
    
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





if __name__ == "__main__":    
    make_prediction(Image.open("..\\data\\model_input_images\\e7.jpg"))
    