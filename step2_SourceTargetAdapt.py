
import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler
from itertools import cycle
import numpy as np
from tqdm import tqdm


#### Implement Step2

def _do_epoch(args,feature_extractor,rot_cls,obj_cls,source_loader,target_loader_train,target_loader_eval,optimizer,device):

    criterion = nn.CrossEntropyLoss()
    feature_extractor.train()
    obj_cls.train()
    rot_cls.train()

    total_source_loader_num = len(source_loader.dataset)
    total_target_loader_train_num = len(target_loader_train.dataset)

    target_loader_train = cycle(target_loader_train)

    img_corrects = 0
    rot_corrects = 0

    for it, (data_source, class_l_source, _, _) in tqdm(enumerate(source_loader)):

        (data_target, _ , data_target_rot, rot_l_target) = next(target_loader_train)

        data_source, class_l_source  = data_source.to(device), class_l_source.to(device)
        data_target, data_target_rot, rot_l_target  = data_target.to(device), data_target_rot.to(device), rot_l_target.to(device)

        optimizer.zero_grad()

        # extract features
        feature_source = feature_extractor(data_source)
        feature_target = feature_extractor(data_target)
        feature_target_rot = feature_extractor(data_target_rot)

        # object prediction
        prediction_source = obj_cls(feature_source)

        # rot prediction
        rot_prediction_target = rot_cls(torch.cat((feature_target, feature_target_rot), dim=1))

        # calculate losses
        class_loss = criterion(prediction_source,class_l_source)
        rot_loss = criterion(rot_prediction_target,rot_l_target)

        loss = class_loss + args.weight_RotTask_step2*rot_loss

        # backward
        loss.backward()

        optimizer.step()

        # calculate accuracy
        _, cls_pred_source = torch.max(prediction_source, 1)
        _, rot_pred = torch.max(rot_prediction_target, 1)

        img_corrects += torch.sum(cls_pred_source == class_l_source)
        rot_corrects += torch.sum(rot_pred == rot_l_target)

    acc_cls = (img_corrects.double() / total_source_loader_num ) * 100
    acc_rot = (rot_corrects.double() / total_source_loader_num) * 100

    # print("#",img_corrects,total_source_loader_num)
    # print("#",rot_corrects,total_target_loader_train_num)
    print("Class Loss %.4f, Class Accuracy %.4f,Rot Loss %.4f, Rot Accuracy %.4f" % (class_loss.item(), acc_cls, rot_loss.item(), acc_rot))

    #### Implement the final evaluation step, computing OS*, UNK and HOS
    feature_extractor.eval()
    obj_cls.eval()
    rot_cls.eval()

    corrects_known = 0
    corrects_unknown = 0
    total_known = 0
    total_unknown = 0

    with torch.no_grad():
        for it, (data, class_l,_,_,_,_) in tqdm(enumerate(target_loader_eval)):
            data, class_l  = data.to(device), class_l.to(device)
            feature = feature_extractor(data)
            prediction = obj_cls(feature)
            _, pred = torch.max(prediction, 1)

            if class_l < args.n_classes_known:
                if pred == class_l:
                    corrects_known += 1
                total_known += 1
            else:
                if pred >= args.n_classes_known:
                    corrects_unknown += 1
                total_unknown += 1
    # print("#",corrects_known,total_known," ",corrects_unknown,total_unknown)
    os_star = corrects_known / total_known
    unk = corrects_unknown / total_unknown
    hos = 2 * os_star * unk / (os_star + unk)
    print(" OS*: %.4f, UNK: %.4f, HOS: %.4f" % (os_star, unk, hos))
    
def step2(args,feature_extractor,rot_cls,obj_cls,source_loader,target_loader_train,target_loader_eval,device):
    optimizer, scheduler = get_optim_and_scheduler(feature_extractor,rot_cls,obj_cls, args.epochs_step2, args.learning_rate, args.train_all)


    for epoch in range(args.epochs_step2):
        print('Epoch: ',epoch)
        _do_epoch(args,feature_extractor,rot_cls,obj_cls,source_loader,target_loader_train,target_loader_eval,optimizer,device)
        scheduler.step()