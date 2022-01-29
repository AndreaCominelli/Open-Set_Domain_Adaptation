#from data_helper import get_train_transformers
import torch.utils.data as data
from PIL import Image
from itertools import permutations
import random
import math
import matplotlib.pyplot as plt
import numpy as npy
from torchvision import transforms

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
def random_jigsaw(img, img_size, dim):
    size = int(dim[0]*dim[1])
    jigsaw_index = random.randint(0,math.factorial(size))

    #img_size must be a multiple of both dim[0] and dim[1]
    (width, height) = (int(img_size / dim[0]), int(img_size / dim[1]))

    pieces = []
    for i in range(size):
        left = (width*i) % (width*dim[0])
        right = left + width
        top = height*(int(i/dim[0]))
        bottom = top + height
        pieces.append(img.crop((left, top, right, bottom)))

    perm = list(permutations(range(size)))

    img = Image.new("RGB", (width*dim[0],height*dim[1]))

    for i in range(size):
        left = (width*i) % (width*dim[0])
        top = height*(int(i/dim[0]))
        img.paste(pieces[perm[jigsaw_index][i]], (left, top))

    return img, jigsaw_index

def some_jigsaw(img, img_size, dim):
    size = int(dim[0]*dim[1])

    #img_size must be a multiple of both dim[0] and dim[1]
    (width, height) = (int(img_size / dim[0]), int(img_size / dim[1]))

    pieces = []
    for i in range(size):
        left = (width*i) % (width*dim[0])
        right = left + width
        top = height*(int(i/dim[0]))
        bottom = top + height
        pieces.append(img.crop((left, top, right, bottom)))

    perm = list(permutations(range(size)))

    img = Image.new("RGB", (width*dim[0],height*dim[1]))

    ### DECIDE WHAT TO DO, IF RETURN 720 IMAGES (CRAZY) OR IF RETURN SOME IMAGES IN THE MIDDLE LIKE NOW...................!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    to_return = []
    split = 4
    step = len(perm) / 6
    for i in range(4):
        for i in range(size):
            left = (width*i) % (width*dim[0])
            top = height*(int(i/dim[0]))
            img.paste(pieces[perm[step*i+step][i]], (left, top))

    return to_return[0], to_return[1], to_return[2], to_return[3]

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
            jig0, jig1, jig2, jig3 = some_jigsaw(img, self.img_size, self.jigsaw_dim)
        
        return img, int(self.labels[index]), img_90, img_180, img_270, flipped, jig0, jig1, jig2, jig3, img_path

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
        [transforms.Resize((222,222)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    source_ds = Dataset(names=names,labels=lbls,path_dataset="./data",img_transformer=transform)

    img, index_img, img_rot, index_rot = source_ds[0]
    imshow(img, index_img)
    imshow(img_rot, index_rot)