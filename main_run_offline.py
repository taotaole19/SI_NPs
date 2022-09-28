
import random
import numpy as np
import torch
import torch.optim as optim


from image_utils.image_metadataset import svhn_metadataset, cifar10_metadataset
from offline_meta_tr_and_te import *
from image_param_list import *
from meta_models import *





def compute_result(param_list, 
                   meta_net, 
                   cat_dim, 
                   check_lvm, 
                   whether_condition, 
                   tr_loss_fun, 
                   te_loss_fun, 
                   tr_hard, 
                   te_hard, 
                   beta0, 
                   beta1, 
                   image_data, 
                   rand_eval, 
                   writer=1):
    
    if image_data == 'CIFAR10':
        train_loader,eval_loader=cifar10_metadataset()
    elif image_data == 'SVHN':
        train_loader,eval_loader=svhn_metadataset()

    args,random,device=param_list()
    random.seed(args.seed)
    
    meta_net=meta_net(args).to(device)
    
    optimizer = optim.Adam(meta_net.parameters(), lr=5e-4)
    
    meta_tr_results, meta_te_nll_results, meta_te_mse_results = run_tr_te(args=args, 
                                                                          meta_net=meta_net, 
                                                                          cat_dim=cat_dim, 
                                                                          net_optim=optimizer, 
                                                                          train_loader=train_loader, 
                                                                          eval_loader=eval_loader, 
                                                                          check_lvm=check_lvm, 
                                                                          whether_condition=whether_condition, 
                                                                          tr_loss_fun=tr_loss_fun, 
                                                                          te_loss_fun=te_loss_fun, 
                                                                          tr_hard=tr_hard, 
                                                                          te_hard=te_hard, 
                                                                          beta0=beta0, 
                                                                          beta1=beta1, 
                                                                          rand_eval=rand_eval, 
                                                                          writer=writer)
    


train_model = 'NP_VEM'

if train_model == 'NP_VEM':
    param_list=params_npvem
    meta_net=NP_VEM
    cat_dim=None
    check_lvm='NP_VEM'     
    whether_condition=False
    tr_loss_fun=None
    te_loss_fun=None
    tr_hard=None
    te_hard=None 
    beta0=None
    beta1=None  



compute_result(param_list, 
               meta_net, 
               cat_dim, 
               check_lvm, 
               whether_condition, 
               tr_loss_fun, 
               te_loss_fun, 
               tr_hard, 
               te_hard, 
               beta0, 
               beta1, 
               image_data='CIFAR10', 
               rand_eval=True, 
               writer=1)
