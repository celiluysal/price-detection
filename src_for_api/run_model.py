from unet import UNet
import torch, cv2, copy
import numpy as np 
from torchvision import transforms as T
from PIL import Image
from PIL import ImageOps

# PARAMETERS
cuda = False
input_shape = (224, 224)
n_classes = 2
PADDING = 0.05

# DIRECTORIES
model_file_name = "model_1"
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
    
    image = ImageOps.exif_transpose(image)
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
    
    cropped_image = crop_price(predict_mask, cv2_image, PADDING)
    
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

def crop_price(mask, image, padding):
    width, height = image.shape[1],image.shape[0]
    mask = cv2.resize(mask, (width, height))

    selected_mask = find_biggest_area(mask)
    x,y,w,h = cv2.boundingRect(selected_mask)
    
    spc_v = int(h * padding)
    spc_h = int(w * padding)
    x1,y1,x2,y2 = x - spc_h, y - spc_v, x + w + spc_h, y + h + spc_v
    x1,y1,x2,y2 = limit_edges(width, height, x1,y1,x2,y2)
    
    crop_image = image[y1:y2, x1:x2]

    if(w or h):
        return crop_image
    else:
        return None
    
def limit_edges(w, h, x1, y1, x2, y2):
    x1 = 0 if x1 < 0 else x1
    y1 = 0 if y1 < 0 else y1
    x2 = w if x2 > w else x2
    y2 = h if y2 > h else y2
    return x1, y1, x2, y2


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
    img = Image.open("..\\data\\model_input_images\\e.jpeg")
    # img = ImageOps.exif_transpose(img)
    # img.show()
    
    image = make_prediction(img)
    cv2.imwrite("..\\data\\model_outputs\\test8_cropped\\e2.jpeg",image)
    # cv2.imshow("fg",image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    
    
    
    
    