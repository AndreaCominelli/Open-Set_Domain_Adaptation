import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler
from itertools import cycle
import numpy as np
from tqdm import tqdm

#### Implement Step2
# In order to solve the second step, I just evaluate together the performance on source and target set
# I pass to my backbone the source image in order to predict its category
# then the target image with its rotated version. Here i must recognise the rotation i applied
# Finally, i combine the 2 losses

def _do_epoch(feature_extractor, obj_cls, self_cls, source_loader,target_loader_train,target_loader_eval,weight,optimizer, n_class_known, n_class_tot, device):

    criterion = nn.CrossEntropyLoss()
    feature_extractor.train()
    self_cls[0].train()
    
    total_source_loader_num = len(source_loader.dataset)
    target_loader_train = cycle(target_loader_train)

    img_corrects = 0
    self_corrects = 0
    
    for _, (data_source, class_l_source, _, _) in tqdm(enumerate(source_loader)):

        ### CHECK!!!!
        # if args.batch_size > target_loader_train batch size, raise an exception
        # it cannot iterate over the target_loader_train batch size
        (data_target, _ , self_data_target, self_l_target) = next(target_loader_train)

        data_source, class_l_source  = data_source.to(device), class_l_source.to(device)
        ### CHECK!!!!
        data_target, self_data_target, self_l_target  = data_target.to(device), self_data_target.to(device), self_l_target.to(device)

        optimizer.zero_grad()

        # extract features
        # i send to the backbone the source image, the target image and the target image with self-supervised transformation
        feature_source = feature_extractor(data_source)
        feature_target = feature_extractor(data_target)
        feature_target_self = feature_extractor(self_data_target)

        # object prediction on the source image
        prediction_source = obj_cls(feature_source)
        _, cls_pred_source = torch.max(prediction_source, 1)

        # training the rotation classifiers, similar to step1
        # prediction on target set of the rotated image (in case of rotation)
        self_prediction_target = self_cls[0](torch.cat((feature_target_self, feature_target), dim=1))
        self_loss = criterion(self_prediction_target, self_l_target)
        _, self_preds = torch.max(self_prediction_target, 1)
        self_corrects += torch.sum(self_preds == self_l_target.data)
        
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
    # This is just an evaluation loop on the target set
    # The model must be able to recognise objects belonging to the target set and 
    # unknown categories.
    feature_extractor.eval()
    obj_cls.eval()
    
    corrects_known = 0
    corrects_unknown = 0
    total_known = 0
    total_unknown = 0

    with torch.no_grad():
        for _, (data, class_l,_,_) in tqdm(enumerate(target_loader_eval)):
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
    return os_star, unk, hos
   
def step2(args,feature_extractor, obj_cls, self_cls, source_loader,target_loader_train,target_loader_eval, weight, n_epochs, device):
    optimizer, scheduler = get_optim_and_scheduler(feature_extractor,obj_cls, self_cls, n_epochs, args.learning_rate, args.weight_decay, args.train_all)
    
    hos_values = {}
    os_star = 0
    unk = 0
    hos = 0

    for epoch in range(n_epochs):
        print('Epoch: ',epoch)
        os_star, unk, hos = _do_epoch(feature_extractor, obj_cls, self_cls, source_loader,target_loader_train,target_loader_eval,weight,optimizer, args.n_classes_known, args.n_classes_tot, device)
        
        hos_values[epoch] = hos
    
        if args.enable_scheduler:    
            scheduler.step()
        
    hos_values_list = [(k, v) for k, v in hos_values.items()]
    
    hos_stats = open(f"./stats/hos_stats_{args.self_sup_task}.txt", "a")

    statistics = ""
    for hos_acc in hos_values_list: # 1:0.05 , 2:0.15... epoch:accuracy
        statistics += str(hos_acc[0]) + ":" + str(hos_acc[1]) + ","
    statistics += str(weight) + "\n"
    hos_stats.write(statistics)
    hos_stats.close()

    metrics = open(f"./models/{args.source}/metrics_{args.self_sup_task}.txt", "a")
    metrics.write(f"{args.self_sup_task} - ({args.source} -- {args.target}) - (OS: {str(round(os_star, 4))}, UNK: {str(round(unk, 4))}, HOS: {str(round(hos, 4))})\n")
    metrics.close()