import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import random
from tqdm import tqdm
from torch import nn

#### Implement the evaluation on the target for the known/unknown separation
def evaluation(args, feature_extractor, self_cls, target_loader_eval, device):

    feature_extractor.eval()
    self_cls[0].eval()
    
    softmax = nn.Softmax(dim=1)

    normality_scores = []
    ground_truth = []
    known_samples = []
    unknown_samples = []

    with torch.no_grad():
        for _, (img ,class_l, img_self_sup, img_path) in tqdm(enumerate(target_loader_eval)):
            img, class_l = img.to(device), class_l.to(device)
            
            if class_l >= args.n_classes_known:
                ground_truth.append(0)
            else:
                ground_truth.append(1)

            normality_score_list = []

            img_out = feature_extractor(img)
            img_prediction = softmax(self_cls[0](torch.cat((img_out, img_out), dim=1)))
            normality_score_list.append(torch.max(img_prediction, 1)[0].item())

            for i in img_self_sup: # img_self_sup contiene tutte le rotazioni dell'immagine img
                im = i.to(device)
                self_out = feature_extractor(im)
                self_prediction = softmax(self_cls[0](torch.cat((self_out, img_out), dim=1)))
                normality_score_list.append(torch.max(self_prediction, 1)[0].item())

            normality_score = np.mean(normality_score_list)
            
            normality_scores.append(normality_score)
            # set the threshold as the mean of the normality scores
            # threshold = np.mean(np.array(normality_scores))

            if normality_score > args.threshold:
                # target samples which labels is under n_class_known
                known_samples.append(img_path[0] + ' ' + str(class_l.item()))
            else:
                # maybe this was the problem: objects that do not belong to the source set
                # will have a label that is grather than n_class_known, thus
                # it would be impossible for the model to predict them, since its last
                # fc layer is composed by n_class_know + 1 neurons (+ 1 for the unknown class)
                if class_l.item() > args.n_classes_known:
                    l = args.n_classes_known
                else:
                    l = class_l.item()
                unknown_samples.append(img_path[0] + ' ' + str(l))
    
    auroc = roc_auc_score(ground_truth, normality_scores)
    print('normality_scores samples:',normality_scores[:5],normality_scores[4360:])
    print('ground_truth samples:',ground_truth[:5],ground_truth[4360:])
    print('AUROC %.4f' % auroc)

    # create new txt files
    rand = random.randint(0,100000)
    print('Generated random number is :', rand)

    # This txt files will have the names of the source images and the names of the target images selected as unknown
    target_unknown = open('new_txt_list/' + args.source + '_known_' + str(rand) + '.txt','a')

    # This txt files will have the names of the target images selected as known
    target_known = open('new_txt_list/' + args.target + '_known_' + str(rand) + '.txt','a')

    for path in known_samples:
        target_known.write(path+"\n")
    target_known.close()

    for path in unknown_samples:
        target_unknown.write(path+"\n")
    target_unknown.close()

    number_of_known_samples = len(known_samples)
    number_of_unknown_samples = len(unknown_samples)

    print('The number of target samples selected as known is: ', number_of_known_samples)
    print('The number of target samples selected as unknown is: ', number_of_unknown_samples)

    return rand
