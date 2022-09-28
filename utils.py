'''
This file is to record some modules in implementations.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import math
import numpy as np



def get_act(act_type):
    '''
    Get activation layer for neural networks.
    '''
    
    if act_type=='ReLU':
        return nn.ReLU()
    elif act_type=='LeakyReLU':
        return nn.LeakyReLU()
    elif act_type=='ELU':
        return nn.ELU()
    elif act_type=='Sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError('Invalid argument for act_type')
    

def set_global_seeds(i):
    '''
    get all random related packages fixed with a seed.
    '''
    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i)
    

class Context_Encoder(nn.Module):
    '''
    Encoder network for [x_c, y_c]
    '''
    def __init__(self, input_size, hidden_size, act_type, num_layers, output_size):
        super(Context_Encoder, self).__init__()
        
        self.emb_c_modules=[]
        self.emb_c_modules.append(nn.Linear(input_size,hidden_size))
        for i in range(num_layers):
            self.emb_c_modules.append(get_act(act_type))
            self.emb_c_modules.append(nn.Linear(hidden_size,hidden_size))
        self.emb_c_modules.append(get_act(act_type))
        self.context_net=nn.Sequential(*self.emb_c_modules)
        
        self.mu_net=nn.Linear(hidden_size, output_size) # map [x_tr,x_te,[x_tr,y_tr]]->z_c to local l.v. z_*
        self.logvar_net=nn.Linear(hidden_size, output_size) # map [x_tr,x_te,[x_tr,y_tr]]->z_t to local l.v. z_*
    
    def forward(self,x,mean_dim=1):
        # input x in the form [x_c,y_c]
        out=self.context_net(x)
        out=torch.mean(out,dim=mean_dim)
        mu, logvar=self.mu_net(out),self.logvar_net(out)
        
        '''
        print (logvar.min())
        print (logvar.max()) 
        '''
        
        return (mu,logvar)    
        


            
        
        
        
 

