
import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler
from tqdm import tqdm

#### Implement Step1

"""def _do_epoch(args,feature_extractor,rot_cls,obj_cls,dataloaders,optimizer,device,phase):
    
    if phase == "train":
        print("TRAINING MODE")
        feature_extractor.train()
        obj_cls.train()
        rot_cls.train()
    else:
        print("VALIDATION MODE")
        feature_extractor.eval()
        obj_cls.eval()
        rot_cls.eval()

    running_img_corrects = 0
    running_rot_corrects = 0

    for i, (imgs, lbls, rot_imgs, rot_lbls) in tqdm(enumerate(dataloaders[phase])):
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        rot_imgs = rot_imgs.to(device)
        rot_lbls = rot_lbls.to(device)

        # forward
        with torch.set_grad_enabled(phase == "train"):
            imgs_out = feature_extractor(imgs)
            imgs_predictions = obj_cls(imgs_out)
            
            rot_out = feature_extractor(rot_imgs)
            rot_predictions = rot_cls(torch.cat((rot_out, imgs_out), dim=1))
            
            _, imgs_preds = torch.max(imgs_predictions, 1)
            _, rot_preds = torch.max(rot_predictions, 1)

            # compute loss

            img_loss = nn.CrossEntropyLoss(imgs_predictions, lbls)
            rot_loss = nn.CrossEntropyLoss(rot_predictions, rot_lbls)

            loss = img_loss + args.weight_RotTask_step1*rot_loss

            if phase == "train":
                    # compute gradient + update params
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # statistics
            # raccolgo in ciascun batch la loss per quel batch e quanti elementi sono stati
            # classificati correttamente

        #running_loss += loss.item() * imgs.size(0)
        running_img_corrects += torch.sum(imgs_preds == lbls.data)
        running_rot_corrects += torch.sum(rot_preds == rot_lbls.data)

    #class_loss = running_loss / dataset_sizes[phase]
    img_acc = (running_img_corrects.double() / len(dataloaders[phase].dataset)) * 100
    rot_acc = (running_rot_corrects.double() / len(dataloaders[phase].dataset)) * 100

    return img_loss, img_acc, rot_loss, rot_acc"""

def _do_epoch(args,feature_extractor,rot_cls,obj_cls, flip_cls, jigsaw_cls, source_loader,optimizer,device,criterion):
    
    feature_extractor.train()
    obj_cls.train()
    rot_cls.train()
    flip_cls.train()
    jigsaw_cls.train()
    
    running_img_corrects = 0
    running_rot_corrects = 0
    running_flip_corrects = 0
    running_jigsaw_corrects = 0

    for i, (imgs, lbls, rot_imgs, rot_lbls, flip_img, flip_labl, jigsaw_img, jigsaw_labl) in tqdm(enumerate(source_loader)):
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        rot_imgs = rot_imgs.to(device)
        rot_lbls = rot_lbls.to(device)
        flip_img = flip_img.to(device)
        flip_labl = flip_labl.to(device)
        jigsaw_img = jigsaw_img.to(device)
        jigsaw_labl = jigsaw_labl.to(device)

        # forward
        imgs_out = feature_extractor(imgs)
        imgs_predictions = obj_cls(imgs_out)

        rot_out = feature_extractor(rot_imgs)
        rot_predictions = rot_cls(torch.cat((rot_out, imgs_out), dim=1))

        flip_out = feature_extractor(flip_img)
        flip_prediction = flip_cls(torch.cat((flip_out, imgs_out), dim=1))

        jigsaw_out = feature_extractor(jigsaw_img)
        jigsaw_prediction = jigsaw_cls(torch.cat((jigsaw_out, imgs_out), dim=1))

        _, imgs_preds = torch.max(imgs_predictions, 1)
        _, rot_preds = torch.max(rot_predictions, 1)
        _, flip_pred = torch.max(flip_prediction, 1)
        _, jigsaw_pred = torch.max(jigsaw_prediction, 1)

        # compute loss

        img_loss = criterion(imgs_predictions, lbls)
        rot_loss = criterion(rot_predictions, rot_lbls)
        flip_loss = criterion(flip_prediction, flip_labl)
        jigsaw_loss = criterion(jigsaw_prediction, jigsaw_labl)

        loss = img_loss + args.weight_RotTask_step1*rot_loss + args.weight_FlipTask_step1*flip_loss + args.weight_JigsawTask_step1*jigsaw_loss

        # compute gradient + update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #statistics
        running_img_corrects += torch.sum(imgs_preds == lbls.data)
        running_rot_corrects += torch.sum(rot_preds == rot_lbls.data)
        running_flip_corrects += torch.sum(flip_pred == flip_labl.data)
        running_jigsaw_corrects += torch.sum(jigsaw_pred == jigsaw_labl.data)

    img_acc = (running_img_corrects.double() / len(source_loader.dataset)) * 100
    rot_acc = (running_rot_corrects.double() / len(source_loader.dataset)) * 100
    flip_acc = (running_flip_corrects.double() / len(source_loader.dataset)) * 100
    jigsaw_acc = (running_jigsaw_corrects.double() / len(source_loader.dataset)) * 100

    return img_loss, img_acc, rot_loss, rot_acc, flip_loss, flip_acc, jigsaw_loss, jigsaw_acc

def step1(args,feature_extractor,rot_cls,obj_cls, flip_cls, jigsaw_cls, source_loader,device):
    optimizer, scheduler = get_optim_and_scheduler(feature_extractor,obj_cls, rot_cls, flip_cls, jigsaw_cls, args.epochs_step1, args.learning_rate, args.train_all)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs_step1):
        print('Epoch: ',epoch)
        class_loss, acc_cls, rot_loss, acc_rot, flip_loss, flip_acc, jigsaw_loss, jigsaw_acc = _do_epoch(args,feature_extractor,rot_cls,obj_cls, flip_cls, jigsaw_cls, source_loader,optimizer,device,criterion)
        print("Class Loss %.4f, Class Accuracy %.4f,Rot Loss %.4f, Rot Accuracy %.4f, Flip Loss %.4f, Flip Accuracy %.4f, Jigsaw Loss %.4f, Jigsaw Accuracy %.4f" % (class_loss.item(),acc_cls,rot_loss.item(), acc_rot, flip_loss.item(), flip_acc, jigsaw_loss.item(), jigsaw_acc))
        scheduler.step()
    
    torch.save(feature_extractor.state_dict(), "./feature_extractor_params.pt")
    torch.save(obj_cls.state_dict(), "./obj_cls_params.pt")
    torch.save(rot_cls.state_dict(), "./rot_cls_params.pt")
    torch.save(flip_cls.state_dict(), "./flip_cls_params.pt")
    torch.save(jigsaw_cls.state_dict(), "./jigsaw_cls_params.pt")
