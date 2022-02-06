import argparse
import os
import math
import datetime
import torch
from Logger import Logger
import sys
import data_helper
from resnet import resnet18_feat_extractor, Classifier

from step1_KnownUnknownSep import step1
from eval_target import evaluation
import numpy as npy

from torch.utils.data.sampler import SubsetRandomSampler

from step2_SourceTargetAdapt import step2


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--source", default='Art', help="Source name")
    parser.add_argument("--target", default='Clipart', help="Target name")
    parser.add_argument("--n_classes_known", type=int, default=45, help="Number of known classes")
    parser.add_argument("--n_classes_tot", type=int, default=65, help="Number of unknown classes")

    # dataset path
    parser.add_argument("--path_dataset", default="./data", help="Path where the Office-Home dataset is located")

    # data augmentation
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--random_grayscale", default=0.1, type=float,help="Randomly greyscale the image")

    # training parameters
    parser.add_argument("--image_size", type=int, default=222, help="Image size")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")

    parser.add_argument("--epochs_step1", type=int, default=10, help="Number of epochs of step1 for known/unknown separation")
    parser.add_argument("--epochs_step2", type=int, default=10, help="Number of epochs of step2 for source-target adaptation")

    parser.add_argument("--train_all", type=bool, default=True, help="If true, all network weights will be trained")

    parser.add_argument("--weight_RotTask_step1", type=float, default=0.5, help="Weight for the rotation loss in step1")
    parser.add_argument("--weight_FlipTask_step1", type=float, default=0.5, help="Weight for the flip loss in step1")
    parser.add_argument("--weight_JigsawTask_step1", type=float, default=0.5, help="Weight for the jigsaw loss in step1")
    parser.add_argument("--weight_RotTask_step2", type=float, default=0.5, help="Weight for the rotation loss in step2")
    parser.add_argument("--weight_FlipTask_step2", type=float, default=0.5, help="Weight for the flip loss in step2")
    parser.add_argument("--weight_JigsawTask_step2", type=float, default=0.5, help="Weight for the jigsaw loss in step2")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for the known/unkown separation")

    parser.add_argument("--jigsaw_dimension", type=tuple, default=(2,2), help="(horizontal_blocks, vertical_blocks)")

    # tensorboard logger
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--folder_name", default=None, help="Used by the logger to save logs")

    #choose different agrument to do mandatory and variation part
    parser.add_argument("--ros_version",default='ROS',help='You can choose naive ROS or varation1 with flip or with jigsaw or varation2')

    return parser.parse_args()


class Trainer:
    def __init__(self, args):
        self.args = args
        if self.args.ros_version not in ['ROS','variation1','variation2']:
            raise ValueError("You can not use a ROS version that is not in 'ROS','variation1','variation2' !")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        
        # initialize the network with a number of classes equals to the number of known classes + 1 (the unknown class, trained only in step2)
        self.feature_extractor = resnet18_feat_extractor()
        self.obj_classifier = Classifier(512,self.args.n_classes_known+1)

        self.feature_extractor = self.feature_extractor.to(self.device)
        self.obj_cls = self.obj_classifier.to(self.device)

        #[ROS, Variation1,Variation2] 
        if self.args.ros_version == 'ROS':
            self.rot_classifier = Classifier(512*2,4)
            self.rot_cls = self.rot_classifier.to(self.device)

        elif self.args.ros_version == 'variation2':
            # initialize (n_classes_known + 1)  classifiers
            # initialize the mutli class rot classifiers
            self.rot_classifiers = [] 
            for _ in range(args.n_classes_known+1):
                self.rot_classifiers.append(Classifier(512*2,4))
            self.rot_cls = []
            
            for i in range(args.n_classes_known+1):
                self.rot_cls.append(self.rot_classifiers[i].to(self.device))

        elif self.args.ros_version == 'variation1':
            # rot
            self.rot_classifier = Classifier(512*2,4)
            self.rot_cls = self.rot_classifier.to(self.device)
            ### flip classifier
            self.flip_classifier = Classifier(512*2,2)
            ### jigsaw classifier 3x3-permutation
            print(math.factorial(args.jigsaw_dimension[0]*args.jigsaw_dimension[1]))
            self.jigsaw_classifier = Classifier(512*2,math.factorial(args.jigsaw_dimension[0]*args.jigsaw_dimension[1]))
            self.flip_cls = self.flip_classifier.to(self.device)
            self.jigsaw_cls = self.flip_classifier.to(self.device)
        
        '''
        ### flip classifier
        self.flip_classifier = Classifier(512*2,2)
        ### jigsaw classifier 3x3-permutation
        print(math.factorial(args.jigsaw_dimension[0]*args.jigsaw_dimension[1]))
        self.jigsaw_classifier = Classifier(512*2,math.factorial(args.jigsaw_dimension[0]*args.jigsaw_dimension[1]))

        self.feature_extractor = self.feature_extractor.to(self.device)
        self.obj_cls = self.obj_classifier.to(self.device)
        self.rot_cls = self.rot_classifier.to(self.device)
        self.flip_cls = self.flip_classifier.to(self.device)
        self.jigsaw_cls = self.jigsaw_classifier.to(self.device)
        '''
        source_path_file = 'txt_list/'+args.source+'_known.txt'
        self.source_loader = data_helper.get_train_dataloader(args,source_path_file)
        target_path_file = 'txt_list/' + args.target + '.txt'
        self.target_loader_train = data_helper.get_val_dataloader(args,target_path_file)
        self.target_loader_eval = data_helper.get_val_dataloader(args,target_path_file)

        print("Source: ",self.args.source," Target: ",self.args.target)
        print("Dataset size: source %d, target %d" % (len(self.source_loader.dataset), len(self.target_loader_train.dataset)))

    def do_training(self):

        # just check if final training parameters are saved somewhere. If so, so not train again

        if not os.path.isfile("./feature_extractor_params.pt") and not os.path.isfile("./rot_cls_params.pt"):
            print('Step 1 --------------------------------------------')
            if self.args.ros_version == 'ROS' or self.args.ros_version == 'variation2':
                step1(self.args,self.feature_extractor,self.rot_cls,self.obj_cls,self.source_loader,self.device)
            elif self.args.ros_version == 'variation1':
                step1_var1(self.args,self.feature_extractor,self.rot_cls,self.obj_cls, self.flip_cls, self.jigsaw_cls, self.source_loader,self.device)
        
        print('Target - Evaluation -- for known/unknown separation')

        # if params are already computed, load the model and procede with its evaluation
        try:
            if self.args.ros_version == 'ROS':
                self.rot_cls.load_state_dict(torch.load("./models/rot_cls_params.pt"), strict=False)
            elif self.args.ros_version == 'variation2':
                for i in range(self.args.n_classes_known):
                    self.rot_cls[i].load_state_dict(torch.load("./models/rot_cls_params_{}.pt".format(i)), strict=False)
            elif self.args.ros_version == 'variation1':
                self.flip_cls.load_state_dict(torch.load("./models/flip_cls_params.pt"), strict=False)
                self.jigsaw_cls.load_state_dict(torch.load("./models/jigsaw_cls_params.pt"), strict=False)
        except FileNotFoundError:
            raise FileNotFoundError("Can not evaluate saved models with wrong ros_version! Please clear the models folder and run again!")

        self.feature_extractor.load_state_dict(torch.load("./models/feature_extractor_params.pt"), strict=False)

        if self.args.ros_version == 'ROS' or self.args.ros_version == 'variation2':
            rand = evaluation(self.args,self.feature_extractor,self.rot_cls,self.target_loader_eval,self.device)
        elif self.args.ros_version == 'variation1':
            raise NotImplementedError
            # add the variation1's evaluation here

        '''
        self.feature_extractor.load_state_dict(torch.load("./feature_extractor_params.pt"), strict=False)
        self.rot_cls.load_state_dict(torch.load("./rot_cls_params.pt"), strict=False)
        self.flip_cls.load_state_dict(torch.load("./flip_cls_params.pt"), strict=False)
        self.jigsaw_cls.load_state_dict(torch.load("./jigsaw_cls_params.pt"), strict=False)

        rand = evaluation(self.args,self.feature_extractor,self.rot_cls, self.flip_cls, self.jigsaw_cls, self.target_loader_eval,self.device)
        ''' 
        # new dataloaders
        source_path_file = 'new_txt_list/' + self.args.source + '_known_'+str(rand)+'.txt'
        self.source_loader = data_helper.get_train_dataloader(self.args,source_path_file)

        target_path_file = 'new_txt_list/' + self.args.target + '_known_' + str(rand) + '.txt'
        self.target_loader_train = data_helper.get_train_dataloader(self.args,target_path_file)
        self.target_loader_eval = data_helper.get_val_dataloader(self.args,target_path_file)

        print('Step 2 --------------------------------------------')
        if self.args.ros_version == 'ROS' or self.args.ros_version == 'variation2':
            step2(self.args,self.feature_extractor,self.rot_cls,self.obj_cls,self.source_loader,self.target_loader_train,self.target_loader_eval,self.device)
        elif self.args.ros_version == 'variation1':
            raise NotImplementedError
            # add the variation1's step2 here
            ###        

def main():
    args = get_args()
    #set up the logger
    current_time = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    sys.stdout = Logger(args.folder_name+'/'+current_time+'.log',sys.stdout)
    sys.stderr = Logger(args.folder_name+'/'+current_time+'.log',sys.stderr)    
    trainer = Trainer(args)
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()