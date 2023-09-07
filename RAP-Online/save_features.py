# ------------------------------------------------------------
# Modified from 'CloserLookFewShot' 
# By: Jie Hong (jie.hong@anu.edu.au)
# Reference: https://github.com/wyharveychen/CloserLookFewShot
# ------------------------------------------------------------
import sys
sys.dont_write_bytecode = True 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import glob
import h5py

import configs
import backbone
from data.datamgr     import SimpleDataManager
from methods.protonet import ProtoNet
from methods.maml     import MAML
from io_utils         import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file 


def save_features(model, data_loader, outfile):

    f = h5py.File(outfile, 'w')
    max_count = len(data_loader)*data_loader.batch_size
    all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
    all_feats=None
    count=0
    for i, (x,y) in enumerate(data_loader):

        if i%10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))
        x = x.cuda()
        x_var = Variable(x)
        feats, _ = model(x_var)

        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
        all_feats[count:count+feats.size(0)]  = feats.data.cpu().numpy()
        all_labels[count:count+feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()


if __name__ == '__main__':
    params = parse_args('save_features')
    assert params.method != 'maml' and params.method != 'maml_approx', 'maml do not support save_feature and run'

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

    split = params.split
    loadfile = configs.data_dir[params.dataset] + split + '.json'

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'
    checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    print(params.save_iter)
    if params.save_iter != -1:
        modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
    else:
        modelfile   = get_best_file(checkpoint_dir)

    if params.save_iter != -1:
        outfile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + "_" + str(params.save_iter)+ ".hdf5") 
    else:
        outfile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + ".hdf5") 

    datamgr     = SimpleDataManager(image_size, batch_size = 60)
    data_loader = datamgr.get_data_loader(loadfile, aug = False)

    if params.method in ['maml']: 
       raise ValueError('MAML do not support save feature')
    else:
        model = model_dict[params.model]()
        model = nn.DataParallel(model)

    model = model.cuda()
    tmp = torch.load(modelfile)
    
    state = tmp['state']
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
            state[newkey] = state.pop(key)
        else:
            state.pop(key)
            
    model.load_state_dict(state)
    model.eval()

    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    save_features(model, data_loader, outfile)
