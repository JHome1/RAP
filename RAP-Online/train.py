# ------------------------------------------------------------
# Modified from 'CloserLookFewShot' 
# By: Jie Hong (jie.hong@anu.edu.au)
# Reference: https://github.com/wyharveychen/CloserLookFewShot
# ------------------------------------------------------------
import sys
sys.dont_write_bytecode = True 
import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import glob

import configs
import backbone
from data.datamgr     import SimpleDataManager, SetDataManager
from methods.protonet import ProtoNet
from methods.maml     import MAML
from io_utils         import model_dict, parse_args, get_resume_file 


def train(base_loader, val_loader, rein_loader, model, optimization, start_epoch, stop_epoch, params):

    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
       raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0       

    for epoch in range(start_epoch, stop_epoch):
        model.train()
        model.train_loop(epoch, base_loader, optimizer, params.checkpoint_dir, rein_loader) # model are called by reference, no need to return 
        model.eval()

        print(params.checkpoint_dir)
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        acc = model.test_loop(val_loader, params.checkpoint_dir)
        if acc > max_acc:
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

    return model


if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    base_file = configs.data_dir[params.dataset] + 'base.json'
    val_file  = configs.data_dir[params.dataset] + 'val.json'

    backbone.policy_y_range      = params.policy_y_range
    backbone.num_execute_actions = params.num_execute_actions
    backbone.PATH_base_model     = params.PATH_base_model

    if 'Conv' in params.model:
        backbone.action_h = 10
        backbone.action_w = 10
        backbone.action_dim = backbone.action_h*backbone.action_w
        backbone.state_dim = 8+64
        backbone.policy_embedding_dim = [256]  

    if 'Conv' in params.model:
        image_size = 84
    else:
        image_size = 224

    optimization = 'Adam'

    if params.stop_epoch == -1: 
        if params.n_shot == 1:
            params.stop_epoch = 1200
        elif params.n_shot == 5:
            params.stop_epoch = 800
        else:
            params.stop_epoch = 600 # default

    if params.method in ['protonet', 'maml']:
        # n_query = max(1, int(16* params.test_n_way/params.train_n_way)) # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        n_query      = params.n_query
        n_query_rein = params.n_query_rein
 
        train_few_shot_params = dict(n_way = params.train_n_way, n_support=params.n_shot) 
        base_datamgr          = SetDataManager(image_size, n_query=n_query,  **train_few_shot_params)
        base_loader           = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

        rein_datamgr          = SetDataManager(image_size, n_query=n_query_rein, **train_few_shot_params)
        rein_loader           = rein_datamgr.get_data_loader(val_file, aug=False)

        test_few_shot_params  = dict(n_way = params.test_n_way, n_support=params.n_shot) 
        val_datamgr           = SetDataManager(image_size, n_query=n_query, **test_few_shot_params)
        val_loader            = val_datamgr.get_data_loader(val_file, aug=False) 
        # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor        

        if params.method == 'protonet':
            model = ProtoNet(model_dict[params.model], **train_few_shot_params)

        elif params.method in ['maml']:
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            model = MAML(model_dict[params.model], approx = (params.method == 'maml_approx') , **train_few_shot_params)
            model = model.cuda()
    else:
       raise ValueError('Unknown method')

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method == 'maml' or params.method == 'maml_approx' :
        # stop_epoch = params.stop_e traipoch * model.n_task # maml use multiple tasks in one update
        stop_epoch = 1200
    print(stop_epoch)

    model = train(base_loader, val_loader, rein_loader, model, optimization, start_epoch, stop_epoch, params)
