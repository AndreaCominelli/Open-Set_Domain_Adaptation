import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler
from tqdm import tqdm

#### Implement Step1

# multi-head = 1 if there is only one head otherwise it will represent the number of clasess (and of heads, one per class)
def _do_epoch(args, feature_extractor, obj_cls, self_sup_type, self_cls, source_loader, weight, optimizer,device,criterion):
    
    feature_extractor.train()
    obj_cls.train()
    self_cls[0].train()
    
    running_img_corrects = 0
    running_self_corrects = 0

    for _, (imgs, lbls, self_imgs, self_lbls) in tqdm(enumerate(source_loader)):
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        self_imgs = self_imgs.to(device)
        self_lbls = self_lbls.to(device)

        # forward
        imgs_out = feature_extractor(imgs)
        imgs_predictions = obj_cls(imgs_out)
        # loss
        img_loss = criterion(imgs_predictions, lbls)
        _, imgs_preds = torch.max(imgs_predictions, 1)
        #statistics
        running_img_corrects += torch.sum(imgs_preds == lbls.data)

        # forward
        # i pick just one image and its random rotated version and i try to predict the transformation
        # try to implement multi-rotation classifier
        self_out = feature_extractor(self_imgs)
        
        self_predictions = self_cls[0](torch.cat((self_out, imgs_out), dim=1))
        # try to implement multi-rotation classifier
        if self_sup_type == "rotation_mh":
            self_lbls = (lbls * 4) + self_lbls
        elif self_sup_type == "flip_mh":
            self_lbls = (lbls * 2) + self_lbls
        elif self_sup_type == "jigsaw_mh":
            self_lbls = (lbls * args.jigsaw_permutations) + self_lbls
            
        self_loss = criterion(self_predictions, self_lbls)
        _, self_preds = torch.max(self_predictions, 1)
        running_self_corrects += torch.sum(self_preds == self_lbls.data)

        # loss
        loss = img_loss + weight*self_loss

        # compute gradient + update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    img_acc = (running_img_corrects.double() / len(source_loader.dataset)) * 100
    self_acc = (running_self_corrects.double() / len(source_loader.dataset)) * 100

    return img_loss, img_acc, self_loss, self_acc

def step1(args, feature_extractor,obj_cls, self_sup_type, self_cls, source_loader, weight, n_epochs, device):
    optimizer, scheduler = get_optim_and_scheduler(feature_extractor,obj_cls, self_cls, n_epochs, args.learning_rate, args.weight_decay, args.train_all)
    criterion = nn.CrossEntropyLoss()
    self_accuracies = {}
    obj_accuracies = {}

    for epoch in range(n_epochs):
        print('Epoch: ',epoch)
        obj_loss, obj_acc, self_loss, self_acc = _do_epoch(args, feature_extractor, obj_cls, self_sup_type, self_cls, source_loader, weight, optimizer,device,criterion)
        
        self_accuracies[epoch] = self_acc
        obj_accuracies[epoch] = obj_acc
        
        print("Obj-Class Loss %.4f, Obj-Class Accuracy %.4f, Self_sup Loss %.4f, Self_sup Accuracy %.4f" % (obj_loss.item(), obj_acc, self_loss.item(), self_acc))
        if args.enable_scheduler:    
            scheduler.step()

    # write all statistics on file. We will plot them later

    self_accuracies_list = [(k, v) for k, v in self_accuracies.items()]
    obj_accuracies_list = [(k, v) for k, v in obj_accuracies.items()]
    
    self_stats = open("./stats/self_stats.txt", "a")
    obj_stats = open("./stats/obj_stats.txt", "a")
    statistics = ""
    for self_acc in self_accuracies_list: # 1:0.05 , 2:0.15... epoch:accuracy
        statistics += str(self_acc[0]) + ":" + str(self_acc[1].item()) + ","
    statistics += str(weight) + "\n"
    self_stats.write(statistics)
    self_stats.close()

    statistics = ""
    for obj_acc in obj_accuracies_list: # 1:0.05 , 2:0.15... epoch:accuracy
        statistics += str(obj_acc[0]) + ":" + str(obj_acc[1].item()) + ","
    statistics += str(weight) + "\n"
    obj_stats.write(statistics)
    obj_stats.close()
    
    return feature_extractor.state_dict(), obj_cls.state_dict(), self_cls[0].state_dict()