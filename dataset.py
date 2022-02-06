#from data_helper import get_train_transformers
import torch.utils.data as data
from PIL import Image
from itertools import permutations
import random
import math
#import matplotlib
#matplotlib.use('TkAgg', force=True)
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

def random_flip(img):
    flip_index = random.randint(0,1)

    if flip_index:
        img = transforms.functional.vflip(img)
    
    return img, flip_index

def flip_2_times(img):
    flipped = transforms.functional.vflip(img)

    return img, flipped

### what about inserting factorial number into the args??
### verify that the image from torch computation is compatible with pillow format....
def random_jigsaw(img, img_size, jigsaw_dim):
    size = int(jigsaw_dim[0]*jigsaw_dim[1])
    jigsaw_index = random.randint(0,math.factorial(size)-1)

    #img_size must be a multiple of both dim[0] and dim[1]
    (width, height) = (int(img_size / jigsaw_dim[0]), int(img_size / jigsaw_dim[1]))

    pieces = []
    for i in range(size):
        left = (width*i) % (width*jigsaw_dim[0])
        top = height*(int(i/jigsaw_dim[0]))
        pieces.append(transforms.functional.crop(img, top, left, height, width))

    perm = list(permutations(range(size)))

    img = torch.empty(0)

    for i in range(jigsaw_dim[1]):
        img_row = torch.empty(0)
        for j in range(jigsaw_dim[0]):
            img_row = torch.cat((img_row, pieces[perm[jigsaw_index][i*jigsaw_dim[0]+j]]), dim=2)
        img = torch.cat((img, img_row), dim=1)

    return img, jigsaw_index

def some_jigsaw(img, img_size, jigsaw_dim, nbr_imgs):
    size = int(jigsaw_dim[0]*jigsaw_dim[1])

    #img_size must be a multiple of both dim[0] and dim[1]
    (width, height) = (int(img_size / jigsaw_dim[0]), int(img_size / jigsaw_dim[1]))

    pieces = []
    for i in range(size):
        left = (width*i) % (width*jigsaw_dim[0])
        top = height*(int(i/jigsaw_dim[0]))
        pieces.append(transforms.functional.crop(img, top, left, height, width))

    perm = list(permutations(range(size)))

    to_return = []
    step = len(perm) / (nbr_imgs+1)
    for n in range(nbr_imgs):
        img = torch.empty(0)
        for i in range(jigsaw_dim[1]):
            img_row = torch.empty(0)
            for j in range(jigsaw_dim[0]):
                img_row = torch.cat((img_row, pieces[perm[step*n+step][i*jigsaw_dim[0]+j]]), dim=2)
            img = torch.cat((img, img_row), dim=1)
        to_return.append(img)

    return to_return

class Dataset(data.Dataset):
    def __init__(self, names, labels, img_size, jigsaw_dim, path_dataset,img_transformer=None):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self.img_size = img_size
        self.jigsaw_dim = jigsaw_dim
        self._image_transformer = img_transformer

    def __getitem__(self, index):

        img_name = self.names[index]    
        img = Image.open(self.data_path +"/"+ img_name)
        
        if self._image_transformer:
            img = self._image_transformer(img)
            img_rot, index_rot = choose_random_rotation(img)
            ###
            img_flip, index_flip = random_flip(img)
            img_jigsaw, index_jigsaw = random_jigsaw(img, self.img_size, self.jigsaw_dim)
        
        return img, int(self.labels[index]), img_rot, index_rot, img_flip, index_flip, img_jigsaw, index_jigsaw

    def __len__(self):
        return len(self.names)

class TestDataset(data.Dataset):
    def __init__(self, names, labels, img_size, jigsaw_dim, path_dataset,img_transformer=None):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self.img_size = img_size
        self.jigsaw_dim = jigsaw_dim
        self._image_transformer = img_transformer

    def __getitem__(self, index):

        img_name = self.names[index]
        img_path = self.data_path +"/"+ img_name
        img = Image.open(img_path)
        
        if self._image_transformer:
            img = self._image_transformer(img)
            _, img_90, img_180, img_270 = rotate_4_times(img)
            ###
            _, flipped = flip_2_times(img)
            jig = some_jigsaw(img, self.img_size, self.jigsaw_dim, 4)
        
        return img, int(self.labels[index]), img_90, img_180, img_270, flipped, jig[0], jig[1], jig[2], jig[3], img_path

    def __len__(self):
        return len(self.names)

#  should we need to add the methods of Dataset and TestData for var1 ?


'''
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
        [transforms.Resize((222,222)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    source_ds = Dataset(names=names,labels=lbls, img_size=222, jigsaw_dim=(3,2), path_dataset="./data",img_transformer=transform)

    img, index_img, img_rot, index_rot, flip_img, flip_labl, jigsaw_img, jigsaw_labl = source_ds[0]
    #imshow(img, index_img)
    #imshow(img_rot, index_rot)
    #imshow(flip_img, flip_labl)
    print(jigsaw_img.shape)
    imshow(jigsaw_img, jigsaw_labl)
'''