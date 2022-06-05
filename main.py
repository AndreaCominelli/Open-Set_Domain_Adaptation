import argparse
from email.policy import strict
import os
import torch

import data_helper
from resnet import resnet18_feat_extractor, Classifier

from step1_KnownUnknownSep import step1
from step2_SourceTargetAdapt import step2
from eval_target import evaluation

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
    parser.add_argument("--random_blur", default=1.5, type=float,help="Randomly blur the image")
    parser.add_argument("--kernel_blur_size", default=21, type=float,help="Set the size of kernel blur")

    # training parameters
    parser.add_argument("--image_size", type=int, default=222, help="Image size (dimension should be compatible with jigsaw dimension)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--enable_scheduler", type=bool, default=False, help="If true, the system will apply a learning rate decay policy every n epochs")
    
    parser.add_argument("--epochs_rot_step1", type=int, default=10, help="Number of epochs of step1 for known/unknown separation in rotation training")
    parser.add_argument("--epochs_rot_step2", type=int, default=10, help="Number of epochs of step2 for source-target adaptation in rotation training")
    parser.add_argument("--epochs_rot_MH_step1", type=int, default=10, help="Number of epochs of step1 for known/unknown separation in rotation multi-head training")
    parser.add_argument("--epochs_rot_MH_step2", type=int, default=10, help="Number of epochs of step2 for source-target adaptation in rotation multi-head training")
    parser.add_argument("--epochs_flip_step1", type=int, default=10, help="Number of epochs of step1 for known/unknown separation in flip training")
    parser.add_argument("--epochs_flip_step2", type=int, default=10, help="Number of epochs of step2 for source-target adaptation in flip training")
    parser.add_argument("--epochs_flip_MH_step1", type=int, default=10, help="Number of epochs of step1 for known/unknown separation in flip multi-head training")
    parser.add_argument("--epochs_flip_MH_step2", type=int, default=10, help="Number of epochs of step2 for source-target adaptation in flip multi-head training")
    parser.add_argument("--epochs_jigsaw_step1", type=int, default=10, help="Number of epochs of step1 for known/unknown separation in jigsaw training")
    parser.add_argument("--epochs_jigsaw_step2", type=int, default=10, help="Number of epochs of step2 for source-target adaptation in jigsaw training")
    parser.add_argument("--epochs_jigsaw_MH_step1", type=int, default=10, help="Number of epochs of step1 for known/unknown separation in jigsaw multi-head training")
    parser.add_argument("--epochs_jigsaw_MH_step2", type=int, default=10, help="Number of epochs of step2 for source-target adaptation in jigsaw multi-head training")

    parser.add_argument("--train_all", type=bool, default=True, help="If true, all network weights will be trained")

    parser.add_argument("--weight_RotTask_step1", type=float, default=0.5, help="Weight for the rotation loss in step1")
    parser.add_argument("--weight_RotTask_MH_step1", type=float, default=0.5, help="Weight for the rotation multi-head loss in step1")
    parser.add_argument("--weight_FlipTask_step1", type=float, default=0.5, help="Weight for the flip loss in step1")
    parser.add_argument("--weight_FlipTask_MH_step1", type=float, default=0.5, help="Weight for the flip multi-head loss in step1")
    parser.add_argument("--weight_JigsawTask_step1", type=float, default=0.5, help="Weight for the jigsaw loss in step1")
    parser.add_argument("--weight_JigsawTask_MH_step1", type=float, default=0.5, help="Weight for the jigsaw multi-head loss in step1")
    parser.add_argument("--weight_RotTask_step2", type=float, default=0.5, help="Weight for the rotation loss in step2")
    parser.add_argument("--weight_RotTask_MH_step2", type=float, default=0.5, help="Weight for the rotation multi-head loss in step2")
    parser.add_argument("--weight_FlipTask_step2", type=float, default=0.5, help="Weight for the flip loss in step2")
    parser.add_argument("--weight_FlipTask_MH_step2", type=float, default=0.5, help="Weight for the flip multi-head loss in step2")
    parser.add_argument("--weight_JigsawTask_step2", type=float, default=0.5, help="Weight for the jigsaw loss in step2")
    parser.add_argument("--weight_JigsawTask_MH_step2", type=float, default=0.5, help="Weight for the jigsaw multi-head loss in step2")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for the known/unkown separation")

    parser.add_argument("--jigsaw_dimension", type=tuple, default=(3,3), help="(horizontal_blocks, vertical_blocks)")
    parser.add_argument("--jigsaw_permutations", type=int, default=31, help="Number of max permutations (maximum Hamming Distance)")

    # tensorboard logger
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--folder_name", default=None, help="Used by the logger to save logs")

    # model options
    parser.add_argument("--save_model", type=bool, default=False, help="If true, the current model will be saved between one training session and the next one")
    parser.add_argument("--self_sup_task", default="Rot", help="The type of self supervised task to apply: Rot, Flip, Jig, RotMH, FlipMH")
    parser.add_argument("--train_dont_save", type=bool, default=False, help="If true, we retrain the model even if another model as already present. if save_model is true, the new model will overwrite the other one")
    
    return parser.parse_args()


class Trainer:
    def __init__(self, args):
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # initialize the network with a number of classes equals to the number of known classes + 1 (the unknown class, trained only in step2)
        self.feature_extractor = resnet18_feat_extractor().to(self.device)
        self.obj_cls = Classifier(512,self.args.n_classes_known + 1).to(self.device)

        self.cls_dict = dict(
            rot_cls = ("rotation", [Classifier(512*2,4).to(self.device)]),
            rot_MH_cls = ("rotation_mh", [Classifier(512*2, (self.args.n_classes_known * 4) + 3).to(self.device)]),
            flip_cls = ("flip", [Classifier(512*2,2).to(self.device)]),
            flip_MH_cls = ("flip_mh", [Classifier(512*2, (self.args.n_classes_known * 2) + 1).to(self.device)]),
            jigsaw_cls = ("jigsaw", [Classifier(512*2,args.jigsaw_permutations).to(self.device)]),
            jigsaw_MH_cls = ("jigsaw_mh", [Classifier(512*2, (self.args.n_classes_known * self.args.jigsaw_permutations) + self.args.jigsaw_permutations - 1).to(self.device)])
        )
        
        ###DO ANOTHER DICT IF WE NEED DIFFERENT LEARNING RATES PER TASK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        self.step1_weights = dict(
            rot_cls = args.weight_RotTask_step1,
            rot_MH_cls = args.weight_RotTask_MH_step1,
            flip_cls = args.weight_FlipTask_step1,
            flip_MH_cls = args.weight_FlipTask_MH_step1,
            jigsaw_cls = args.weight_JigsawTask_step1,
            jigsaw_MH_cls = args.weight_JigsawTask_MH_step1
        )

        self.step1_epochs = dict(
            rot_cls = args.epochs_rot_step1,
            rot_MH_cls = args.epochs_rot_MH_step1,
            flip_cls = args.epochs_flip_step1,
            flip_MH_cls = args.epochs_flip_MH_step1,
            jigsaw_cls = args.epochs_jigsaw_step1,
            jigsaw_MH_cls = args.epochs_jigsaw_MH_step1
        )

        self.step2_weights = dict(
            rot_cls = args.weight_RotTask_step2,
            rot_MH_cls = args.weight_RotTask_MH_step2,
            flip_cls = args.weight_FlipTask_step2,
            flip_MH_cls = args.weight_FlipTask_MH_step2,
            jigsaw_cls = args.weight_JigsawTask_step2,
            jigsaw_MH_cls = args.weight_JigsawTask_MH_step2
        )

        self.step2_epochs = dict(
            rot_cls = args.epochs_rot_step2,
            rot_MH_cls = args.epochs_rot_MH_step2,
            flip_cls = args.epochs_flip_step2,
            flip_MH_cls = args.epochs_flip_MH_step2,
            jigsaw_cls = args.epochs_jigsaw_step2,
            jigsaw_MH_cls = args.epochs_jigsaw_MH_step2
        )

        self.source_path_file = 'txt_list/'+args.source+'_known.txt'
        self.target_path_file = 'txt_list/' + args.target + '.txt'
    
        with open(self.target_path_file, "r") as f:
            target_list_known_size = len(list(filter(lambda path: int(path.rstrip().split()[1]) < int(self.args.n_classes_known), f.readlines())))


        print("Source: ",self.args.source," Target: ",self.args.target)
        print(f"Number of known samples in target set: {target_list_known_size}")

    def do_training(self, self_sup_cls):
        print(self.args)
        self.current_sup_cls = self.cls_dict[self_sup_cls]

        self.source_loader = data_helper.get_train_dataloader(self.args,self.source_path_file, self.current_sup_cls[0])
        self.target_loader_train = data_helper.get_val_dataloader(self.args,self.target_path_file, self.current_sup_cls[0])
        self.target_loader_eval = data_helper.get_val_dataloader(self.args,self.target_path_file, self.current_sup_cls[0])
        print("Dataset size: source %d, target %d" % (len(self.source_loader.dataset), len(self.target_loader_train.dataset)))
 
        # just check if final training parameters are saved somewhere. If so, not train again unless train_dont_save is true
    
        if (not os.path.isfile(f"./models/{self.args.source}/{self.args.self_sup_task}/feature_extractor_params.pt") and not os.path.isfile(f"./models/{self.args.source}/{self.args.self_sup_task}/obj_cls_params.pt")) or self.args.train_dont_save : # and not model_present:
            print('Step 1 --------------------------------------------')
            fe_model, obj_model, self_model = step1(self.args, self.feature_extractor, self.obj_cls, self.current_sup_cls[0], self.current_sup_cls[1], self.source_loader, self.step1_weights[self_sup_cls], self.step1_epochs[self_sup_cls], self.device)
            
            if self.args.save_model:
                torch.save(fe_model, f"./models/{self.args.source}/{self.args.self_sup_task}/feature_extractor_params.pt")
                torch.save(obj_model, f"./models/{self.args.source}/{self.args.self_sup_task}/obj_cls_params.pt")
                torch.save(self_model, f"./models/{self.args.source}/{self.args.self_sup_task}/self_{self_sup_cls}_cls_params.pt")
                
        # if params are already computed, load the model and procede with its evaluation
        
        if self.args.save_model:
            self.feature_extractor.load_state_dict(torch.load(f"./models/{self.args.source}/{self.args.self_sup_task}/feature_extractor_params.pt"), strict=False)
            self.obj_cls.load_state_dict(torch.load(f"./models/{self.args.source}/{self.args.self_sup_task}/obj_cls_params.pt"), strict=False)
            self.current_sup_cls[1][0].load_state_dict(torch.load(f"./models/{self.args.source}/{self.args.self_sup_task}/self_{self_sup_cls}_cls_params.pt"), strict=False)
        
        print('Target - Evaluation -- for known/unknown separation')
        rand = evaluation(self.args, self.feature_extractor, self.current_sup_cls[1], self.target_loader_eval, self.device)
        
        # we need the standard self_classifier for step 2
        if self_sup_cls == "rot_MH_cls":
            self.current_sup_cls = self.cls_dict["rot_cls"]
        elif self_sup_cls == "flip_MH_cls":
            self.current_sup_cls = self.cls_dict["flip_cls"]
        elif self_sup_cls == "jigsaw_MH_cls":
            self.current_sup_cls = self.cls_dict["jigsaw_cls"]

        # new dataloaders
        # source_path_file -> contains the names of source and target images selected as unknown
        # In this way i can train a model which is capable of recognising what are the known categories
        # and the unknown ones        
        source_path_file = 'new_txt_list/' + self.args.source + '_known_'+str(rand)+'.txt'
        self.source_loader = data_helper.get_train_dataloader(self.args,source_path_file, self.current_sup_cls[0])


        # target_path_file -> contains the names of target images which are considered as known
        target_path_file = 'new_txt_list/' + self.args.target + '_known_' + str(rand) + '.txt'
        self.target_loader_train = data_helper.get_train_dataloader(self.args,target_path_file, self.current_sup_cls[0])
        self.target_loader_eval = data_helper.get_val_dataloader(self.args,target_path_file, self.current_sup_cls[0])

        print('Step 2 --------------------------------------------')    
        step2(self.args, self.feature_extractor, self.obj_cls, self.current_sup_cls[1], self.source_loader, self.target_loader_train, self.target_loader_eval, self.step2_weights[self_sup_cls], self.step2_epochs[self_sup_cls], self.device)
    
        
def main():
    args = get_args()
    trainer = Trainer(args)
    if args.self_sup_task == "Rot":
        print("---Rotation Self-Supervised Task---")
        trainer.do_training("rot_cls")
    elif args.self_sup_task == "Flip":
        print("---Flip Self-Supervised Task---")
        trainer.do_training("flip_cls")
    elif args.self_sup_task == "Jig":
        print("---Jigsaw Self-Supervised Task---")
        trainer.do_training("jigsaw_cls")
    
    elif args.self_sup_task == "RotMH":
        print("---Multi-Head Rotation Self-Supervised Task---")
        trainer.do_training("rot_MH_cls")
    elif args.self_sup_task == "FlipMH":
        print("---Multi-Head Flip Self-Supervised Task---")
        trainer.do_training("flip_MH_cls")
    elif args.self_sup_task == "JigMH":
        print("---Multi-Head Jigsaw Self-Supervised Task---")
        trainer.do_training("jigsaw_MH_cls")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
