from matplotlib import image
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import random
from tqdm import tqdm
from torch import nn

#### Implement the evaluation on the target for the known/unknown separation
def evaluation(feature_extractor, self_cls, multi_head, n_classes_known, threshold, target_loader_eval, source_dir, target_dir, device):

    feature_extractor.eval()
    for self_cls_i in self_cls:
        self_cls_i.eval()
    
    softmax = nn.Softmax(dim=1)

    normality_scores = []
    ground_truth = []
    known_samples = []
    unknown_samples = []

    with torch.no_grad():
        for _, (img ,class_l, img_self_sup, img_path) in tqdm(enumerate(target_loader_eval)):
            img, class_l = img.to(device), class_l.to(device)
            
            if class_l >= n_classes_known:
                ground_truth.append(0)
            else:
                ground_truth.append(1)

            normality_score_list = []

            if multi_head == 1:
              img_out = feature_extractor(img)
              img_prediction = softmax(self_cls[0](torch.cat((img_out, img_out), dim=1)))
              normality_score_list.append(torch.max(img_prediction, 1)[0].item())

              for i in img_self_sup:
                  im = i.to(device)
                  self_out = feature_extractor(im)
                  self_prediction = softmax(self_cls[0](torch.cat((self_out, img_out), dim=1)))
                  normality_score_list.append(torch.max(self_prediction, 1)[0].item())

              normality_score = np.mean(normality_score_list)
            
            else:
                normality_score = 0
                for i in range(n_classes_known):
                    img_out = feature_extractor(img)
                    img_prediction = softmax(self_cls[i](torch.cat((img_out, img_out), dim=1)))
                    normality_score_list.append(torch.max(img_prediction, 1)[0].item())
                    for im in img_self_sup:
                        im = im.to(device)
                        self_out = feature_extractor(im)
                        self_prediction = softmax(self_cls[i](torch.cat((self_out, img_out), dim=1)))
                        normality_score_list.append(torch.max(self_prediction, 1)[0].item())
                    act_norm_score = np.mean(normality_score_list)
                    if normality_score < act_norm_score:
                        normality_score = act_norm_score

            normality_scores.append(normality_score)

            if normality_score > threshold:
                print(f"IMAGE PATH ----> {img_path[0] + ' ' + str(class_l)}")
                known_samples.append(img_path[0] + ' ' + str(class_l))
            else:
                print(f"IMAGE PATH ----> {img_path[0] + ' ' + str(class_l)}")
                unknown_samples.append(img_path[0] + ' ' + str(class_l))
    
    auroc = roc_auc_score(ground_truth, normality_scores)
    print('normality_scores samples:',normality_scores[:5],normality_scores[4360:])
    print('ground_truth samples:',ground_truth[:5],ground_truth[4360:])
    print('AUROC %.4f' % auroc)

    # create new txt files
    rand = random.randint(0,100000)
    print('Generated random number is :', rand)

    # This txt files will have the names of the source images and the names of the target images selected as unknown
    target_unknown = open('new_txt_list/' + source_dir + '_known_' + str(rand) + '.txt','a')

    # This txt files will have the names of the target images selected as known
    target_known = open('new_txt_list/' + target_dir + '_known_' + str(rand) + '.txt','a')

    for path in known_samples:
        target_known.write(path[0]+"\n")
    target_known.close()

    for path in unknown_samples:
        target_unknown.write(path[0]+"\n")
    target_unknown.close()

    number_of_known_samples = len(known_samples)
    number_of_unknown_samples = len(unknown_samples)

    print('The number of target samples selected as known is: ', number_of_known_samples)
    print('The number of target samples selected as unknown is: ', number_of_unknown_samples)

    return rand
