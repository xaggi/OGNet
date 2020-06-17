from __future__ import print_function
import numpy as np
from sklearn import metrics
import re
lastNum = re.compile(r'(?:[^\d]*(\d+)[^\d]*)+')
from dataloader import load_data
from model import OGNet
from opts import parse_opts

# Test code for the CVPR 2020 paper ->  Old is Gold: Redefining the Adversarially Learned One-Class Classifier Training Paradigm
# https://arxiv.org/abs/2004.07657
# http://openaccess.thecvf.com/content_CVPR_2020/html/Zaheer_Old_Is_Gold_Redefining_the_Adversarially_Learned_One-Class_Classifier_Training_CVPR_2020_paper.html
# Code written by Zaigham and Jin-ha.
def check_auc(g_model_path, d_model_path, opt, i):
    opt.batch_shuffle = False
    opt.drop_last = False
    dataloader = load_data(opt)
    model = OGNet(opt, dataloader)
    model.cuda()
    d_results, labels = model.test_patches(g_model_path, d_model_path, i)
    d_results = np.concatenate(d_results)
    labels = np.concatenate(labels)
    fpr1, tpr1, thresholds1 = metrics.roc_curve(labels, d_results, pos_label=1)  # (y, score, positive_label)
    fnr1 = 1 - tpr1
    eer_threshold1 = thresholds1[np.nanargmin(np.absolute((fnr1 - fpr1)))]
    eer_threshold1 = eer_threshold1
    d_f1 = np.copy(d_results)
    d_f1[d_f1 >= eer_threshold1] = 1
    d_f1[d_f1 < eer_threshold1] = 0
    f1_score = metrics.f1_score(labels, d_f1, pos_label=0)
    print("AUC: {0}, F1_score: {1}".format(metrics.auc(fpr1,tpr1), f1_score))



if __name__ == '__main__':
    opt = parse_opts()
    opt.data_path = './data/test/'  #test data path
    g_model_path = './models/phase_two_g' #generator model path
    d_model_path = './models/phase_two_d' #discriminator model path
    print('working on :', g_model_path, d_model_path)
    check_auc(g_model_path, d_model_path, opt, 0)

