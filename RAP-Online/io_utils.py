# ------------------------------------------------------------
# Modified from 'CloserLookFewShot' 
# By: Jie Hong (jie.hong@anu.edu.au)
# Reference: https://github.com/wyharveychen/CloserLookFewShot
# ------------------------------------------------------------
import torch
import numpy as np
import os
import glob
import argparse
import backbone

model_dict = dict(  
                  Conv4_rein = backbone.Conv4_rein,
                  Conv6_rein = backbone.Conv6_rein,
                  )


def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--dataset'     , default='miniImagenet', help='CUB/miniImagenet')
    parser.add_argument('--model'       , default='Conv4',        help='model: Conv_rein{4|6} / ResNet_rein{10|18}') # 50 and 101 are not used in the paper
    parser.add_argument('--method'      , default='baseline',     help='baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/maml{_approx}') # relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
    parser.add_argument('--train_n_way' , default=5, type=int,    help='class num to classify for training') # baseline and baseline++ would ignore this parameter
    parser.add_argument('--test_n_way'  , default=5, type=int,    help='class num to classify for testing (validation)') # baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--n_shot'      , default=1, type=int,    help='number of labeled data in each class, same as n_support') # baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--train_aug'   , action='store_true',    help='perform data augmentation or not during training') # still required for save_features.py and test.py to find the model path correctly
    parser.add_argument('--n_query_rein', default=4,  type=int,   help='number of rein query data in each class')
    parser.add_argument('--n_query'     , default=16, type=int,   help='number of query data in each class')
    parser.add_argument('--n_task'      , default=4,  type=int,   help='number of task for maml')

    #########################
    # parameters of RAP model
    parser.add_argument('--PATH-base-model',  default='./pre_model/miniImagenet/conv4_1shot_base_model.tar')
    parser.add_argument('--loss-train-beta',  default=1.0)
    parser.add_argument('--loss-policy-beta', default=1.0)
    parser.add_argument('--reward-alpha',     default=1e-4)
    parser.add_argument('--policy-y-range',      nargs='+', type=float, default=[0.5, 1.5])     
    parser.add_argument('--num-execute-actions', type=int,  default=5)

    if script == 'train':
        parser.add_argument('--num_classes' , default=200, type=int, help='total number of classes in softmax, only used in baseline') # make it larger than the maximum label value in base class
        parser.add_argument('--save_freq'   , default=100, type=int,  help='Save frequency')
        parser.add_argument('--start_epoch' , default=0, type=int,   help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=-1, type=int,  help ='Stopping epoch') # for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
        parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
        parser.add_argument('--warmup'      , action='store_true', help='continue from baseline, neglected if resume is true') # never used in the paper

    elif script == 'save_features':
        parser.add_argument('--split'       , default='novel',      help='base/val/novel') # default novel, but you can also test base/val class accuracy if you want 
        parser.add_argument('--save_iter'   , default=-1, type=int, help ='save feature from the model trained in x epoch, use the best model if x is -1')

    elif script == 'test':
        parser.add_argument('--split'       , default='novel',      help='base/val/novel') # default novel, but you can also test base/val class accuracy if you want 
        parser.add_argument('--save_iter'   , default=-1, type=int, help ='saved feature from the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--adaptation'  , action='store_true',  help='further adaptation in test time or not')

    else:
       raise ValueError('Unknown script')
        
    return parser.parse_args()


def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):    
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
