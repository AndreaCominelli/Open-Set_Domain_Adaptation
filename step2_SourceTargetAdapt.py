import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler
from itertools import cycle
import numpy as np
from tqdm import tqdm

#### Implement Step2

def _do_epoch(feature_extractor, obj_cls, self_cls, multi_head, source_loader,target_loader_train,target_loader_eval,weight,optimizer, n_class_known, n_class_tot, device):

    criterion = nn.CrossEntropyLoss()
    feature_extractor.train()
    for self_cls_i in self_cls:
        self_cls_i.train()
    
    total_source_loader_num = len(source_loader.dataset)
    target_loader_train = cycle(target_loader_train)

    img_corrects = 0
    self_corrects = 0

    for it, (data_source, class_l_source, _, _) in tqdm(enumerate(source_loader)):

        ### CHECK!!!!
        (data_target, _ , self_data_target, self_l_target) = next(target_loader_train)

        data_source, class_l_source  = data_source.to(device), class_l_source.to(device)
        ### CHECK!!!!
        data_target, self_data_target, self_l_target  = data_target.to(device), self_data_target.to(device), self_l_target.to(device)

        optimizer.zero_grad()

        # extract features
        feature_source = feature_extractor(data_source)
        feature_target = feature_extractor(data_target)
        feature_target_self = feature_extractor(self_data_target)

        # object prediction
        prediction_source = obj_cls(feature_source)
        _, cls_pred_source = torch.max(prediction_source, 1)

        # training the rotation classifiers, similar to step1
        if multi_head == 1:
            self_prediction_target = self_cls[0](torch.cat((feature_target_self, feature_target), dim=1))
            self_loss = criterion(self_prediction_target, self_l_target)
            _, self_preds = torch.max(self_predictions, 1)
            self_corrects += torch.sum(self_preds == self_l_target.data)
        else:
            self_loss = 0
            for index,class_l in enumerate(class_l_source.int()):
                self_predictions = torch.reshape(self_cls[class_l](torch.cat((feature_target_self[index], feature_target[index]), dim=0)),(1,4))
                self_loss += criterion(self_predictions, torch.reshape(self_l_target[index],(-1,)))
                _, self_preds = torch.max(self_predictions, 1)
                self_corrects += torch.sum(self_preds == self_l_target[index])

        # calculate losses
        class_loss = criterion(prediction_source,class_l_source)
        loss = class_loss + weight*self_loss

        # backward
        loss.backward()

        optimizer.step()

        # calculate accuracy
        img_corrects += torch.sum(cls_pred_source == class_l_source)
        

    acc_cls = (img_corrects.double() / total_source_loader_num ) * 100
    acc_rot = (self_corrects.double() / total_source_loader_num) * 100

    print("Class Loss %.4f, Class Accuracy %.4f,Rot Loss %.4f, Rot Accuracy %.4f" % (class_loss.item(), acc_cls, self_loss.item(), acc_rot))

    #### Implement the final evaluation step, computing OS*, UNK and HOS
    feature_extractor.eval()
    obj_cls.eval()
    
    for rot_cls_i in self_cls:
        rot_cls_i.eval()

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

            if class_l < n_class_known:
                if pred == class_l:
                    corrects_known += 1
                total_known += 1
            else:
                if pred >= n_class_known:
                    corrects_unknown += 1
                total_unknown += 1
    print("#",corrects_known,total_known," ",corrects_unknown,total_unknown)
    os_star = corrects_known / total_known
    unk = corrects_unknown / total_unknown
    hos = 2 * os_star * unk / (os_star + unk)
    print(" OS*: %.4f, UNK: %.4f, HOS: %.4f" % (os_star, unk, hos))

    
def step2(args,feature_extractor, obj_cls, self_cls, multi_head, source_loader,target_loader_train,target_loader_eval, weight, n_epochs, learning_rate, train_all, n_class_known, n_class_tot, device):
    optimizer, scheduler = get_optim_and_scheduler(feature_extractor,obj_cls, self_cls, n_epochs, learning_rate, args.weight_decay, train_all)
    
    # WARNING! NOW WE MUST CHOOSE THE NUM OF EPOCHS BASED ON THE SELF SUPERVISED CLASSIFIER WE CHOOSE

    for epoch in range(args.epochs_rot_step2):
        print('Epoch: ',epoch)
        _do_epoch(feature_extractor, obj_cls, self_cls, multi_head, source_loader,target_loader_train,target_loader_eval,weight,optimizer, n_class_known, n_class_tot, device)
        if args.enable_scheduler:    
            scheduler.step()