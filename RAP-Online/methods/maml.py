# -----------------------------------------------------
# Modified from 'MAML-Pytorch' and 'pytorch-maml'
# By: Jie Hong (jie.hong@anu.edu.au)
# Reference: https://github.com/dragen1860/MAML-Pytorch 
# and https://github.com/katerakelly/pytorch-maml
# -----------------------------------------------------
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
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


class MAML(MetaTemplate):

    def __init__(self, model_func, n_way, n_support, approx = False):
        super(MAML, self).__init__(model_func,  n_way, n_support, change_way = False)

        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
        self.classifier = nn.DataParallel(self.classifier).cuda()
        self.classifier.module.bias.data.fill_(0)
        
        # self.n_task = 4
        self.n_task = params.n_task
        self.task_update_num = 5
        self.train_lr = 0.01
        self.approx = approx # first order approx.        


    def forward(self,x):
        out, log_pi = self.feature.module.forward(x)
        scores      = self.classifier.module.forward(out)
        return scores, log_pi


    def set_forward(self, x, is_feature = False):
        assert is_feature == False, 'MAML do not support fixed feature'
        x = x.cuda() 
        # x = x.to(params.device)
        x_var = Variable(x)
        x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view(self.n_way* self.n_support, *x.size()[2:]) # support data 
        x_b_i = x_var[:,self.n_support:,:,:,:].contiguous().view(self.n_way* self.n_query,   *x.size()[2:]) # query data
        y_a_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).cuda()       # label for support data
        # y_a_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).to(params.device)

        fast_parameters = list(self.parameters()) # the first gradient calcuated in line 45 is based on original weight
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()

        for task_step in range(self.task_update_num):
            scores, _ = self.forward(x_a_i)
            set_loss  = self.loss_fn(scores, y_a_i) 
            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True, allow_unused=True) # build full graph support gradient of gradient
            if self.approx:
                grad = [ g.detach()  for g in grad ] # do not calculate gradient of gradient if using first order approximation
            fast_parameters = []
            for k, weight in enumerate(self.parameters()):
                # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py 
                if weight.fast is None:
                    weight.fast = weight - self.train_lr * grad[k]      # create weight.fast 
                else:
                    if type(grad[k]) == type(None):
                       weight.fast = weight.fast
                    else: weight.fast = weight.fast - self.train_lr * grad[k]  # create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast 
                fast_parameters.append(weight.fast)                            # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts

        scores, log_pi = self.forward(x_b_i)
        return scores, log_pi


    def set_forward_adaptation(self,x, is_feature = False): # overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')


    def set_forward_loss(self, x):
        scores, log_pi = self.set_forward(x, is_feature = False)
        y_b_i = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_query))).cuda()
        # y_b_i = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_query))).to(params.device)
        loss = self.loss_fn(scores, y_b_i)

        return loss, log_pi


    def train_loop(self, epoch, train_loader, optimizer, checkpoint_dir, rein_loader=None): #overwrite parrent function
        print_freq = 10
        avg_loss      = 0
        avg_loss_rein = 0
        task_count = 0
        loss_all = []
        loss_r_all = []
        optimizer.zero_grad()

        # train
        # for i, (x,_) in enumerate(train_loader):
        for i, batch in enumerate(zip(train_loader, cycle(rein_loader))):

            x      = batch[0][0]
            x_rein = batch[1][0]

            # arrange loss
            # self.n_query = x.size(1) - self.n_support
            self.n_query = params.n_query
            assert self.n_way  ==  x.size(0), "MAML do not support way change"
            
            loss, _ = self.set_forward_loss(x)
            avg_loss = avg_loss+loss.item()
            loss_all.append(loss)

            # arrange loss_rein
            self.n_query = params.n_query_rein
            loss_r, log_pi = self.set_forward_loss(x_rein)
            avg_loss_rein = avg_loss_rein + loss_r.item()
            loss_r_all.append(loss_r)

            task_count += 1

            if task_count == self.n_task: # MAML update several tasks at one time
                loss_q   = torch.stack(loss_all).sum(0)
                loss_r_q = torch.stack(loss_r_all).sum(0)

                R = -self.reward_alpha*loss_r_q
                R = R.repeat(1, params.num_execute_actions)

                log_pi = torch.stack(log_pi).transpose(1, 0)
                loss_rein = torch.sum(-log_pi*R, dim=1)
                loss_rein = torch.mean(loss_rein, dim=0)                
                
                loss_joint = self.loss_train_beta*loss_q + self.loss_policy_beta*loss_rein
                loss_joint.backward(retain_graph=True)

                optimizer.step()
                task_count = 0
                loss_all = []
                loss_r_all = []

            optimizer.zero_grad()
            if i % print_freq==0:
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
           
           
    def test_loop(self, test_loader, checkpoint_dir, return_std = False): #overwrite parrent function
        correct =0
        count = 0
        acc_all = []
        
        iter_num = len(test_loader) 
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "MAML do not support way change"
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this/ count_this *100 )

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

        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean
