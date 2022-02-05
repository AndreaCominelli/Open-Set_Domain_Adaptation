
from matplotlib import image
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import random
from tqdm import tqdm

#### Implement the evaluation on the target for the known/unknown separation

def evaluation(args,feature_extractor,rot_cls, flip_cls, jigsaw_cls, target_loader_eval,device):

    feature_extractor.eval()
    rot_cls.eval()
    flip_cls.eval()
    jigsaw_cls.eval()
    
    normality_scores = []
    ground_truth = []
    known_samples = []
    unknown_samples = []

    with torch.no_grad():
        for it, (img ,class_l, img_90, img_180, img_270, flipped, jig0, jig1, jig2, jig3, img_path) in tqdm(enumerate(target_loader_eval)):
            img, class_l = img.to(device), class_l.to(device)
            img_90, img_180, img_270 = img_90.to(device), img_180.to(device), img_270.to(device)
            flipped, jig0, jig1, jig2, jig3 = flipped.to(device), jig0.to(device), jig1.to(device), jig2.to(device), jig3.to(device)
            
            if class_l > args.n_classes_known:
                ground_truth.append(0)
            else:
                ground_truth.append(1)

            img_out   = feature_extractor(img)
            rot_out_90  = feature_extractor(img_90)
            rot_out_180 = feature_extractor(img_180)
            rot_out_270 = feature_extractor(img_270)
            ###
            flip_out = feature_extractor(flipped)
            jig0_out = feature_extractor(jig0)
            jig1_out = feature_extractor(jig1)
            jig2_out = feature_extractor(jig2)
            jig3_out = feature_extractor(jig3)

            rot_predictions_0   = rot_cls(torch.cat((img_out, img_out), dim=1))
            rot_predictions_90  = rot_cls(torch.cat((rot_out_90, img_out), dim=1))
            rot_predictions_180 = rot_cls(torch.cat((rot_out_180, img_out), dim=1))
            rot_predictions_270 = rot_cls(torch.cat((rot_out_270, img_out), dim=1))
            ###
            flip_prediction_img = flip_cls(torch.cat((img_out, img_out), dim=1))
            flip_prediction_flip = flip_cls(torch.cat((flip_out, img_out), dim=1))
            ### some jigsaw samples
            jig_prediction_img = jigsaw_cls(torch.cat((img_out, img_out), dim=1))
            jig_prediction_0 = jigsaw_cls(torch.cat((jig0_out, img_out), dim=1))
            jig_prediction_1 = jigsaw_cls(torch.cat((jig1_out, img_out), dim=1))
            jig_prediction_2 = jigsaw_cls(torch.cat((jig2_out, img_out), dim=1))
            jig_prediction_3 = jigsaw_cls(torch.cat((jig3_out, img_out), dim=1))

            rot_normality_score_0, _   = torch.max(rot_predictions_0, 1)
            rot_normality_score_90, _  = torch.max(rot_predictions_90, 1)
            rot_normality_score_180, _ = torch.max(rot_predictions_180, 1)
            rot_normality_score_270, _ = torch.max(rot_predictions_270, 1)
            ###
            flip_normality_score_img, _ = torch.max(flip_prediction_img, 1)
            flip_normality_score_flip, _ = torch.max(flip_prediction_flip, 1)
            ### some jigsaw samples
            jig_normality_score_img, _ = torch.max(jig_prediction_img, 1)
            jig_normality_score_0, _ = torch.max(jig_prediction_0, 1)
            jig_normality_score_1, _ = torch.max(jig_prediction_1, 1)
            jig_normality_score_2, _ = torch.max(jig_prediction_2, 1)
            jig_normality_score_3, _ = torch.max(jig_prediction_3, 1)

            normality_score = np.mean([rot_normality_score_0.item(), rot_normality_score_90.item(), rot_normality_score_180.item(), rot_normality_score_270.item()])

            normality_scores.append(normality_score)

            if normality_score > args.threshold:
                known_samples.append(img_path)
            else:
                unknown_samples.append(img_path)
    
    auroc = roc_auc_score(ground_truth, normality_scores)
    print('AUROC %.4f' % auroc)

    # create new txt files
    rand = random.randint(0,100000)
    print('Generated random number is :', rand)

    # This txt files will have the names of the source images and the names of the target images selected as unknown
    target_unknown = open('new_txt_list/' + args.source + '_known_' + str(rand) + '.txt','a')

    # This txt files will have the names of the target images selected as known
    target_known = open('new_txt_list/' + args.target + '_known_' + str(rand) + '.txt','a')

    for path in known_samples:
        target_known.write(path[0])
    target_known.close()

    for path in unknown_samples:
        target_unknown.write(path[0])
    target_unknown.close()

    number_of_known_samples = len(known_samples)
    number_of_unknown_samples = len(unknown_samples)

    print('The number of target samples selected as known is: ',number_of_known_samples)
    print('The number of target samples selected as unknown is: ', number_of_unknown_samples)

    return rand






