# ------------------------------------------------------------
# Modified from 'CloserLookFewShot' 
# By: Jie Hong (jie.hong@anu.edu.au)
# Reference: https://github.com/wyharveychen/CloserLookFewShot
# ------------------------------------------------------------
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import utils
from abc import abstractmethod
from itertools import cycle

import os
import scipy.io as io
import matplotlib 
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from io_utils import parse_args
params = parse_args('train')

curve_train_loss = []
curve_rein_loss  = []
curve_val_accur  = []


class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, change_way=True):
        super(MetaTemplate, self).__init__()
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = -1 # (change depends on input)
        self.feature    = model_func()
        self.feat_dim   = self.feature.basenet.final_feat_dim
        self.change_way = change_way  # some methods allow different_way classification during training and test

        # if params.method in ['protonet']:
        self.feature  = nn.DataParallel(self.feature).cuda()
        self.feat_dim = self.feature.module.basenet.final_feat_dim

        self.reward_alpha     = params.reward_alpha
        self.loss_train_beta  = params.loss_train_beta
        self.loss_policy_beta = params.loss_policy_beta

    @abstractmethod
    def set_forward(self, x, is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self, x):
        out, log_pi = self.feature.forward(x)
        return out, log_pi

    def parse_feature(self, x, is_feature):
        x = Variable(x.cuda())
        if is_feature:
            z_all  = x
            log_pi = 0
        else:
            x             = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            z_all, log_pi = self.feature.forward(x)
            z_all         = z_all.view(self.n_way, self.n_support + self.n_query, -1)

        z_support         = z_all[:, :self.n_support]
        z_query           = z_all[:, self.n_support:]

        return z_support, z_query, log_pi

    def correct(self, x):       
        scores, _ = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, optimizer, checkpoint_dir, rein_loader=None):
        print_freq = 10

        avg_loss = 0
        avg_loss_rein = 0 

        # for i, (x, _) in enumerate(train_loader):
        for i, batch in enumerate(zip(train_loader, cycle(rein_loader))):

            x      = batch[0][0]
            x_rein = batch[1][0]

            # arrange loss
            # self.n_query = x.size(1) - self.n_support
            self.n_query = params.n_query
            if self.change_way:
                self.n_way  = x.size(0)
            loss, _ = self.set_forward_loss(x)

            # arrange loss_rein  
            self.n_query = params.n_query_rein
            if self.change_way:
                self.n_way  = x_rein.size(0)        
            loss_r, log_pi = self.set_forward_loss(x_rein)
            log_pi = torch.stack(log_pi).transpose(1, 0)

            R = -self.reward_alpha*loss_r
            R = R.repeat(1, params.num_execute_actions)

            loss_rein = torch.sum(-log_pi*R, dim=1)
            loss_rein = torch.mean(loss_rein, dim=0)
            
            # optimize
            loss_joint = self.loss_train_beta*loss + self.loss_policy_beta*loss_rein
            optimizer.zero_grad()
            loss_joint.backward()
            optimizer.step()

            avg_loss      = avg_loss+loss.item()
            avg_loss_rein = avg_loss_rein+loss_rein.item()

            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Loss_rein {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1), avg_loss_rein/float(i+1)))

        # plot curve
        curve_train_loss.append(avg_loss/float(i+1))
        curve_rein_loss.append(avg_loss_rein/float(i+1))

        io.savemat(os.path.join(checkpoint_dir, 'train_loss.mat'),  {'train_loss':  np.array(curve_train_loss)})
        io.savemat(os.path.join(checkpoint_dir, 'rein_loss.mat'),   {'rein_loss':  np.array(curve_rein_loss)})

        plt.title('train_loss')
        plt.plot(np.arange(len(curve_train_loss)), curve_train_loss)
        plt.savefig(os.path.join(checkpoint_dir, 'train_loss.jpg'))
        plt.close()

        plt.title('rein_loss')
        plt.plot(np.arange(len(curve_rein_loss)), curve_rein_loss)
        plt.savefig(os.path.join(checkpoint_dir, 'rein_loss.jpg'))
        plt.close()

    def test_loop(self, test_loader, checkpoint_dir, record = None):
        correct =0
        count = 0
        acc_all = []
        
        iter_num = len(test_loader) 
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this/ count_this*100)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

        if checkpoint_dir != None:
            # plot curve
            curve_val_accur.append(acc_mean)

            io.savemat(os.path.join(checkpoint_dir, 'val_accur.mat'),  {'val_accur':  np.array(curve_val_accur)})

            plt.title('val_accur')
            plt.plot(np.arange(len(curve_val_accur)), curve_val_accur)
            plt.savefig(os.path.join(checkpoint_dir, 'val_accur.jpg'))
            plt.close()

        return acc_mean

    def set_forward_adaptation(self, x, is_feature = True): #further adaptation, default is fixing feature and train a new softmax clasifier
        assert is_feature == True, 'Feature is fixed in further adaptation'
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous().view(self.n_way* self.n_support, -1)
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1)

        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        y_support = Variable(y_support.cuda())

        linear_clf = nn.Linear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()
        
        batch_size = 4
        support_size = self.n_way* self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy(rand_id[i: min(i+batch_size, support_size) ]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id] 
                scores = linear_clf(z_batch)
                loss = loss_function(scores,y_batch)
                loss.backward()
                set_optimizer.step()

        scores = linear_clf(z_query)
        return scores
