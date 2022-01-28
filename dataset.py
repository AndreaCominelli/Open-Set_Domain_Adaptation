#from data_helper import get_train_transformers
import torch.utils.data as data
from PIL import Image
from random import random
import random
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as npy
from torchvision import transforms
import torch

def _dataset_info(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels

def choose_random_rotation(img):
    rot_index = random.randint(0, 3)
    
    if rot_index == 1: # ruota di 90 gradi
        img = transforms.functional.rotate(img, 90)
    if rot_index == 2: #ruota di 180 gradi
        img = transforms.functional.rotate(img, 180)
    if rot_index == 3: #ruota di 270 gradi
        img = transforms.functional.rotate(img, 270)
    
    return img, rot_index

def rotate_4_times(img):
    img_90 = transforms.functional.rotate(img, 90)
    img_180 = transforms.functional.rotate(img, 180)
    img_270 = transforms.functional.rotate(img, 270)

    return img, img_90, img_180, img_270

class Dataset(data.Dataset):
    def __init__(self, names, labels, path_dataset,img_transformer=None):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer

    def __getitem__(self, index):

        img_name = self.names[index]    
        img = Image.open(self.data_path +"/"+ img_name)
        
        if self._image_transformer:
            img = self._image_transformer(img)
            img_rot, index_rot = choose_random_rotation(img)
        
        return img, int(self.labels[index]), img_rot, index_rot

    def __len__(self):
        return len(self.names)



class TestDataset(data.Dataset):
    def __init__(self, names, labels, path_dataset,img_transformer=None):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer

    def __getitem__(self, index):

        img_name = self.names[index]
        img_path = self.data_path +"/"+ img_name
        img = Image.open(img_path)
        
        if self._image_transformer:
            img = self._image_transformer(img)
            _, img_90, img_180, img_270 = rotate_4_times(img)
        
        return img, int(self.labels[index]), img_90, img_180, img_270, img_path

    def __len__(self):
        return len(self.names)

def imshow(img, lbl):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(npy.transpose(npimg, (1, 2, 0)))
    plt.xlabel(lbl)
    plt.show()

if __name__ == "__main__":
    source_file = 'txt_list/Clipart.txt'
    names, lbls = _dataset_info(source_file)
    
    transform = transforms.Compose(
        [transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    source_ds = Dataset(names=names,labels=lbls,path_dataset="./data",img_transformer=transform)

    img, index_img, img_rot, index_rot = source_ds[0]
    imshow(img, index_img)
    imshow(img_rot, index_rot)