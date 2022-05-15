#from data_helper import get_train_transformers
import torch.utils.data as data
from PIL import Image
from itertools import permutations
import random
import math
import matplotlib.pyplot as plt
import numpy as npy
from torchvision import transforms
import torch
from sklearn.metrics import DistanceMetric
import os

def get_permutations(jigsaw_dim, n_perm):
    size = int(jigsaw_dim[0]*jigsaw_dim[1])
    perm_all = list(permutations(range(size)))
    perm_out = []
    index = random.randint(0,math.factorial(size)-1)
    for _ in range(n_perm):
        perm_out.append(perm_all[index])
        perm_all.pop(index)
        P_a = npy.array(perm_out)  #chosen
        P_b = npy.array(perm_all)  #others

        ham_dist = DistanceMetric.get_metric("hamming").pairwise(P_a, P_b)
        ham_dist = npy.sum(ham_dist, axis=0)
        index = ham_dist.argmax()
    
    return perm_out

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

def all_rotations(img):
    img_rots = [transforms.functional.rotate(img, 90), transforms.functional.rotate(img, 180), transforms.functional.rotate(img, 270)]

    return img_rots

def random_flip(img):
    flip_index = random.randint(0,1)

    if flip_index:
        img = transforms.functional.vflip(img)
    
    return img, flip_index

def flip_img(img):
    flipped = [transforms.functional.vflip(img)]

    return flipped

def random_jigsaw(img, img_size, jigsaw_dim, perm_list):
    size = int(jigsaw_dim[0]*jigsaw_dim[1])
    jigsaw_index = random.randint(0, len(perm_list)-1)

    (width, height) = (int(img_size / jigsaw_dim[0]), int(img_size / jigsaw_dim[1]))

    pieces = []
    for i in range(size):
        left = (width*i) % (width*jigsaw_dim[0])
        top = height*(int(i/jigsaw_dim[0]))
        pieces.append(transforms.functional.crop(img, top, left, height, width))

    img = torch.empty(0)

    for i in range(jigsaw_dim[1]):
        img_row = torch.empty(0)
        for j in range(jigsaw_dim[0]):
            img_row = torch.cat((img_row, pieces[perm_list[jigsaw_index][i*jigsaw_dim[0]+j]]), dim=2)
        img = torch.cat((img, img_row), dim=1)

    return img, jigsaw_index

def all_jigsaw(img, img_size, jigsaw_dim, perm_list):
    size = int(jigsaw_dim[0]*jigsaw_dim[1])

    (width, height) = (int(img_size / jigsaw_dim[0]), int(img_size / jigsaw_dim[1]))

    pieces = []
    for i in range(size):
        left = (width*i) % (width*jigsaw_dim[0])
        top = height*(int(i/jigsaw_dim[0]))
        pieces.append(transforms.functional.crop(img, top, left, height, width))

    to_return = []
    for n in range(len(perm_list)):
        img = torch.empty(0)
        for i in range(jigsaw_dim[1]):
            img_row = torch.empty(0)
            for j in range(jigsaw_dim[0]):
                img_row = torch.cat((img_row, pieces[perm_list[n][i*jigsaw_dim[0]+j]]), dim=2)
            img = torch.cat((img, img_row), dim=1)
        to_return.append(img)

    return to_return

# I pass as a parameter also the self-supervised classifier in order to
# decide what transformation i need to apply

class Dataset(data.Dataset):
    def __init__(self, args, names, labels, self_sup_cls, img_transformer=None):
        self.data_path = args.path_dataset
        self.names = names
        self.labels = labels
        self.img_size = args.image_size
        self.jigsaw_dim = args.jigsaw_dimension
        self.perm_list = get_permutations(args.jigsaw_dimension, args.jigsaw_permutations)
        self._image_transformer = img_transformer
        self.self_sup_cls = self_sup_cls

        #clean data 
        for name in self.names:
            if not os.path.isfile(self.data_path + "/" + name):
                index = self.names.index(name)
                self.names.pop(index)
                self.labels.pop(index)

    def __getitem__(self, index):

        img_name = self.names[index]
        img = Image.open(self.data_path +"/"+ img_name)
            
        if self._image_transformer:
            img = self._image_transformer(img)

        if self.self_sup_cls == "rotation" or self.self_sup_cls == "rotation_mh":
            img_self_sup, index_self_sup = choose_random_rotation(img)
        elif self.self_sup_cls == "flip" or self.self_sup_cls == "flip_mh":
            img_self_sup, index_self_sup = random_flip(img)
        elif self.self_sup_cls == "jigsaw" or self.self_sup_cls == "jigsaw_mh":
            img_self_sup, index_self_sup = random_jigsaw(img, self.img_size, self.jigsaw_dim, self.perm_list)
        else:
            img_self_sup, index_self_sup = img, int(self.labels[index])
        return img, int(self.labels[index]), img_self_sup, index_self_sup

    def __len__(self):
        return len(self.names)

class TestDataset(data.Dataset):
    def __init__(self, args, names, labels, self_sup_cls, img_transformer=None):
        self.data_path = args.path_dataset
        self.names = names
        self.labels = labels
        self.img_size = args.image_size
        self.jigsaw_dim = args.jigsaw_dimension
        self.n_perm = args.jigsaw_permutations
        self._image_transformer = img_transformer
        self.self_sup_cls = self_sup_cls
        self.perm_list = get_permutations(args.jigsaw_dimension, args.jigsaw_permutations)

        #clean data 
        for name in self.names:
            if not os.path.isfile(self.data_path + "/" + name):
                index = self.names.index(name)
                self.names.pop(index)
                self.labels.pop(index)

    def __getitem__(self, index):

        img_name = self.names[index]
        img = Image.open(self.data_path +"/"+ img_name)
            
        if self._image_transformer:
            img = self._image_transformer(img)

        if self.self_sup_cls == "rotation" or self.self_sup_cls == "rotation_mh":
            imgs_self_sup = all_rotations(img)
        elif self.self_sup_cls == "flip" or self.self_sup_cls == "flip_mh":
            imgs_self_sup = flip_img(img)
        elif self.self_sup_cls == "jigsaw" or self.self_sup_cls == "jigsaw_mh":
            imgs_self_sup = all_jigsaw(img, self.img_size, self.jigsaw_dim, self.perm_list)
        else:
            imgs_self_sup = [img]
        return img, int(self.labels[index]), imgs_self_sup, img_name

    def __len__(self):
        return len(self.names)
