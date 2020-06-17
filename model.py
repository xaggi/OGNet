import torch
from torch import nn
import numpy as np
from network import d_net, g_net
from opts import parse_opts
from dataloader import load_data
from sklearn import metrics

def check_auc(g_model_path, d_model_path, i):
    opt_auc = parse_opts()
    opt_auc.batch_shuffle = False
    opt_auc.drop_last = False
    opt_auc.data_path = './data/test/'
    dataloader = load_data(opt_auc)
    model = OGNet(opt_auc, dataloader)
    model.cuda()
    d_results, labels = model.test_patches(g_model_path, d_model_path, i)
    d_results = np.concatenate(d_results)
    labels = np.concatenate(labels)
    fpr1, tpr1, thresholds1 = metrics.roc_curve(labels, d_results, pos_label=1)  # (y, score, positive_label)
    fnr1 = 1 - tpr1
    eer_threshold1 = thresholds1[np.nanargmin(np.absolute((fnr1 - fpr1)))]
    EER1 = fpr1[np.nanargmin(np.absolute((fnr1 - fpr1)))]
    d_f1 = np.copy(d_results)
    d_f1[d_f1 >= eer_threshold1] = 1
    d_f1[d_f1 < eer_threshold1] = 0
    f1_score = metrics.f1_score(labels, d_f1, pos_label=0)
    print("AUC: {0}, EER: {1}, EER_thr: {2}, F1_score: {3}".format(metrics.auc(fpr1,tpr1), EER1,
                                                                  eer_threshold1,f1_score))

class OGNet(nn.Module):
    @staticmethod
    def name():
        return 'Old is Gold: Redefining the adversarially learned one-class classification paradigm'

    def __init__(self, opt, dataloader):
        super(OGNet, self).__init__()
        self.dataloader = dataloader
        self.g = g_net().cuda()
        self.d = d_net().cuda()

    def test_patches(self,g_model_path, d_model_path,i):  #test all patches present inside a folder on given g and d model. Returns d score of each patch
        #input g_model_path, d_model_path. Mostly data path and every other necessary thing it takes from the opts.
        g_checkpoint = torch.load(g_model_path)
        self.g.load_state_dict(g_checkpoint['g_model_state_dict'])
        d_checkpoint = torch.load(d_model_path)
        self.d.load_state_dict(d_checkpoint['d_model_state_dict'])
        self.g.eval()
        self.d.eval()
        labels = []
        d_results = []
        count = 0
        for input, label in self.dataloader:
            input = input.cuda()
            g_output = self.g(input)
            d_fake_output = self.d(g_output)
            count +=1
            d_results.append(d_fake_output.cpu().detach().numpy())
            labels.append(label)
        return d_results, labels

