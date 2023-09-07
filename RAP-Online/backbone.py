# ------------------------------------------------------------------
# Modified from 'CloserLookFewShot' and 'recurrent-visual-attention'
# By: Jie Hong (jie.hong@anu.edu.au)
# Reference: https://github.com/wyharveychen/CloserLookFewShot
# and https://github.com/kevinzakka/recurrent-visual-attention
# ------------------------------------------------------------------
import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
from torch.distributions import Normal

####################
# RAP model settings
action_h = 10
action_w = 10
action_dim = action_h*action_w
state_dim = 8+64
policy_embedding_dim = [256]    
policy_y_range       = [0.0, 1.0]
std = 0.01 
num_execute_actions = 5


# basic resNet model
def init_layer(L):
    # initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class distLinear(nn.Module):

    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True         # see the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) # split the weight update component to direction and norm      

        if outdim <=200:
            self.scale_factor = 2;  # a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github 
        else:
            self.scale_factor = 10; # in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) # matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist) 

        return scores


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):        
        return x.view(x.size(0), -1)


class Linear_fw(nn.Linear): # used in MAML to forward input with fast weight

    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None # lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) # weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out


class Conv2d_fw(nn.Conv2d): # used in MAML to forward input with fast weight

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out

            
class BatchNorm2d_fw(nn.BatchNorm2d): # used in MAML to forward input with fast weight 

    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        # running_mean = torch.zeros(x.data.size()[1]).to(device)
        # running_var  = torch.ones(x.data.size()[1]).to(device)

        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training = True, momentum = 1)
            # batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training = True, momentum = 1)
        return out


# simple conv block
class ConvBlock(nn.Module):
    maml = False # Default

    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        if self.maml:
            self.C      = Conv2d_fw(indim, outdim, 3, padding = padding)
            self.BN     = BatchNorm2d_fw(outdim)
        else:
            self.C      = nn.Conv2d(indim, outdim, 3, padding= padding)
            self.BN     = nn.BatchNorm2d(outdim)
        self.relu   = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool   = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)


    def forward(self,x):
        out = self.trunk(x)
        return out


# simple resNet block
class SimpleBlock(nn.Module):
    maml = False # Default

    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = BatchNorm2d_fw(outdim)
            self.C2 = Conv2d_fw(outdim, outdim,kernel_size=3, padding=1,bias=False)
            self.BN2 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = nn.BatchNorm2d(outdim)
            self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = BatchNorm2d_fw(outdim)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out


# bottleneck block
class BottleneckBlock(nn.Module):
    maml = False # Default
    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlock, self).__init__()
        bottleneckdim = int(outdim/4)
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, bottleneckdim, kernel_size=1,  bias=False)
            self.BN1 = BatchNorm2d_fw(bottleneckdim)
            self.C2 = Conv2d_fw(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
            self.BN2 = BatchNorm2d_fw(bottleneckdim)
            self.C3 = Conv2d_fw(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1,  bias=False)
            self.BN1 = nn.BatchNorm2d(bottleneckdim)
            self.C2 = nn.Conv2d(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
            self.BN2 = nn.BatchNorm2d(bottleneckdim)
            self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = nn.BatchNorm2d(outdim)

        self.relu = nn.ReLU()
        self.parametrized_layers = [self.C1, self.BN1, self.C2, self.BN2, self.C3, self.BN3]
        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, stride=2 if half_res else 1, bias=False)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, stride=2 if half_res else 1, bias=False)

            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):

        short_out = x if self.shortcut_type == 'identity' else self.shortcut(x)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        out = out + short_out

        out = self.relu(out)
        return out


############################
class policy_net(nn.Module):

    def __init__(self, state_dim, action_dim, policy_embedding_dim, policy_y_range, std, num_layer=1, hid_dim=8):
        super(policy_net, self).__init__()

        self.std = std
        self.policy_y_range = policy_y_range

        self.conv = []
        for i in range(num_layer):
            indim = 3 if i == 0 else hid_dim
            outdim = hid_dim
            B = ConvBlock(indim, outdim)
            self.conv.append(B)
        self.conv.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.conv = nn.Sequential(*self.conv)     
 
        self.fc_actor = nn.Sequential(
                                      nn.Linear(state_dim, policy_embedding_dim[0]),
                                      nn.BatchNorm1d(policy_embedding_dim[0]),
                                      nn.ReLU(),
                                      nn.Linear(policy_embedding_dim[0], action_dim),
                                      )

    def forward(self, x, s2):

        # cat state
        s1 = self.conv(x)
        s1 = s1.view(s1.size(0), -1)
        s2 = s2.squeeze(0)

        s = torch.cat([s1, s2], 1)

        # bound a between [self.policy_y_range[0], self.policy_y_range[1]]  
        mu = self.policy_y_range[0] + (self.policy_y_range[1] - self.policy_y_range[0]) * torch.sigmoid(self.fc_actor(s))

        # linear layer        
        noise = torch.zeros_like(mu)
        noise.data.normal_(std=self.std)

        a = mu + noise
        log_pi = Normal(mu, self.std).log_prob(a)
        log_pi = torch.sum(log_pi, dim=1)

        return log_pi, a


#########################
class ConvNet(nn.Module):

    def __init__(self, depth, flatten = True, split_layer=3):
        super(ConvNet, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i <4 )) # only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        # split layer to receive action
        self.trunk_former = nn.Sequential(*list(self.trunk.children())[: split_layer])
        self.trunk_later  = nn.Sequential(*list(self.trunk.children())[split_layer: -1])
        self.trunk_final  = nn.Sequential(*list(self.trunk.children())[-1: ])

        self.average = nn.AdaptiveAvgPool2d((1, 1))
        self.final_feat_dim = 1600

    def forward(self, x, action=1):

        out = self.trunk_former(x)
        out = out*action
        
        out   = self.trunk_later(out)
        state = self.average(out)
        state = torch.flatten(state, 1)

        out = self.trunk_final(out)

        return out, state


###########################
class Conv_rein(nn.Module):

    def __init__(self, action_h, action_w, state_dim, action_dim, policy_embedding_dim, policy_y_range, std, num_execute_actions, depth):
        super(Conv_rein, self).__init__()

        self.policy_net = policy_net(state_dim, action_dim, policy_embedding_dim, policy_y_range, std)

        if depth == 4:
            self.basenet = Conv4()
            tmp = torch.load(PATH_base_model)
            self.basenet.load_state_dict(tmp['state'])

        elif depth == 6:
            self.basenet = Conv6()
            tmp = torch.load(PATH_base_model)
            self.basenet.load_state_dict(tmp['state'])

    def forward(self, x):

        _, s = self.basenet(x)
        log_pi = []

        for i in range(0, num_execute_actions):         

            p, a = self.policy_net(x, s.squeeze(0).detach())

            # reshape the action to attention feature map
            a    = a.view(a.shape[0], action_h, action_w).unsqueeze(1)
            y, s = self.basenet(x, a)

            # record RL samples
            log_pi.append(p)
          
        return y, log_pi


def Conv4():
    return ConvNet(4)


def Conv6():
    return ConvNet(6)


def ResNet10(flatten = True):
    return ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten, split_layer=7)


def ResNet18(flatten = True):
    return ResNet(SimpleBlock, [2,2,2,2],[64,128,256,512], flatten, split_layer=9)


#################
def Conv4_rein():
    return Conv_rein(action_h, action_w, state_dim, action_dim, policy_embedding_dim, policy_y_range, std, num_execute_actions, depth=4) 


#################
def Conv6_rein():
    return Conv_rein(action_h, action_w, state_dim, action_dim, policy_embedding_dim, policy_y_range, std, num_execute_actions, depth=6)
