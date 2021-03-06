
import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler
from tqdm import tqdm

#### Implement Step1
def _do_epoch(args,feature_extractor,rot_cls,obj_cls, flip_cls, jigsaw_cls, source_loader,optimizer,device,criterion):
    
    feature_extractor.train()
    obj_cls.train()
    if args.ros_version == 'variation2':
        for rot_cls_i in rot_cls:
            rot_cls_i.train()
    else:        
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
        ##flip
        flip_out = feature_extractor(flip_img)
        jigsaw_out = feature_extractor(jigsaw_img)
        if args.ros_version == 'variation2':
            rot_loss = 0
            # 对于每张图片，只训练该图片对应类别的旋转角度分类器
            for index,class_l in enumerate(lbls.int()):
                rot_predictions = torch.reshape(rot_cls[class_l](torch.cat((rot_out[index], imgs_out[index]), dim=0)),(1,4))
                rot_loss += criterion(rot_predictions, torch.reshape(rot_lbls[index],(-1,)))
                _, rot_preds = torch.max(rot_predictions, 1)
                running_rot_corrects += torch.sum(rot_preds == rot_lbls[index])
        else:
            rot_predictions = rot_cls(torch.cat((rot_out, imgs_out), dim=1))
            _, rot_preds = torch.max(rot_predictions, 1)
            rot_loss = criterion(rot_predictions, rot_lbls)
            running_rot_corrects += torch.sum(rot_preds == rot_lbls.data)
            ##
            flip_prediction = flip_cls(torch.cat((flip_out, imgs_out), dim=1))
            _, flip_pred = torch.max(flip_prediction, 1)
            jigsaw_prediction = jigsaw_cls(torch.cat((jigsaw_out, imgs_out), dim=1))
            _, jigsaw_pred = torch.max(jigsaw_prediction, 1)

        _, imgs_preds = torch.max(imgs_predictions, 1)        

        '''
        rot_predictions = rot_cls(torch.cat((rot_out, imgs_out), dim=1))

        flip_out = feature_extractor(flip_img)
        flip_prediction = flip_cls(torch.cat((flip_out, imgs_out), dim=1))

        jigsaw_out = feature_extractor(jigsaw_img)
        jigsaw_prediction = jigsaw_cls(torch.cat((jigsaw_out, imgs_out), dim=1))



        _, imgs_preds = torch.max(imgs_predictions, 1)
        _, rot_preds = torch.max(rot_predictions, 1)
        _, flip_pred = torch.max(flip_prediction, 1)
        _, jigsaw_pred = torch.max(jigsaw_prediction, 1)
        '''
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

    if args.ros_version == 'variation2':
        for i in range(args.n_classes_known):
            torch.save(rot_cls[i].state_dict(), "./models/rot_cls_params_{}.pt".format(i))
    else:
        torch.save(rot_cls.state_dict(), "./models/rot_cls_params.pt")
        torch.save(flip_cls.state_dict(), "./flip_cls_params.pt")
        torch.save(jigsaw_cls.state_dict(), "./jigsaw_cls_params.pt")
