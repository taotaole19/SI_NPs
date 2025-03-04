
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from image_utils.image_metadataset import *



##########################################################################################################################
    # Training image completion models in meta learning set-up
##########################################################################################################################
    
    
def train_meta_net(epoch, 
                   meta_net, 
                   cat_dim, 
                   net_optim, 
                   train_loader,
                   check_lvm, 
                   whether_condition, 
                   tr_loss_fun, 
                   tr_hard, 
                   beta0, 
                   beta1):
    
    meta_net.train()
    epoch_train_nll = []
    
    for batch_idx, (y_all, _) in enumerate(train_loader):
        batch_size = y_all.shape[0]
        y_all = y_all.permute(0,2,3,1).contiguous().view(batch_size, -1, 3).cuda()
        
        N = random.randint(1, 1023)  
        idx = get_context_idx(N) 
        idx_list = idx.tolist()
        idx_all = np.arange(1024).tolist()
        x_c = idx_to_x(idx, batch_size)
        y_c = idx_to_y(idx, y_all)
        idx_all_tensor = torch.tensor(idx_all,dtype=torch.long).cuda()
        x = idx_to_x(idx_all_tensor, batch_size).cuda()
        y = idx_to_y(idx_all_tensor, y_all).cuda()
        
        pred_idx = torch.tensor(list(set(idx_all)-set(idx_list)), dtype=torch.long).cuda() 
        x_t = idx_to_x(pred_idx, batch_size).cuda() 
        y_t = idx_to_y(pred_idx, y_all).cuda() 
        
        net_optim.zero_grad()
                
        if check_lvm == 'NP_VEM':
            iw_nll, proposal_nll, iw_nll_evidence=meta_net(x_c,y_c,x,y,x,y)
            loss, b_avg_nll = iw_nll+proposal_nll, iw_nll_evidence          
        else:
            raise NotImplementedError()           
            
        loss.backward()
        net_optim.step()

        epoch_train_nll.append(b_avg_nll.data.cpu())

    avg_tr_nll = np.array(epoch_train_nll).sum() / len(train_loader)

    return avg_tr_nll   


def eval_meta_net(epoch, 
                  meta_net, 
                  eval_loader, 
                  check_lvm, 
                  whether_condition, 
                  te_loss_fun, 
                  te_hard, 
                  num_c_points=None):
    
    meta_net.eval()
    epoch_test_nll = []
    epoch_test_mse = []
    
    with torch.no_grad():
        for batch_idx, (y_all, _) in enumerate(eval_loader):
            batch_size = y_all.shape[0]
            y_all = y_all.permute(0,2,3,1).contiguous().view(batch_size, -1, 3).cuda()
            
            if num_c_points == None:
                N = random.randint(1, 1024)  
            else:
                N = num_c_points

            idx = get_context_idx(N, order_pixels=False) 
            idx_list = idx.tolist()
            idx_all = np.arange(1024).tolist()
            x_c = idx_to_x(idx, batch_size)
            y_c = idx_to_y(idx, y_all)
            idx_all_tensor = torch.tensor(idx_all,dtype=torch.long).cuda()
            x = idx_to_x(idx_all_tensor, batch_size).cuda()
            y = idx_to_y(idx_all_tensor, y_all).cuda()
            
            pred_idx = torch.tensor(list(set(idx_all)-set(idx_list)), dtype=torch.long).cuda() 
            x_t = idx_to_x(pred_idx, batch_size).cuda() 
            y_t = idx_to_y(pred_idx, y_all).cuda() 
            
            if check_lvm == 'NP_VEM':
                mu, logvar, b_avg_nll, y_mean=meta_net.conditional_predict(x_c,y_c,x,y)
                
                b_avg_mse=F.mse_loss(y_mean,y)
                
            else:
                raise NotImplementedError()
            
            epoch_test_nll.append(b_avg_nll.cpu())
            epoch_test_mse.append(b_avg_mse.cpu())

    avg_te_nll = np.array(epoch_test_nll).sum() /len(eval_loader)
    avg_te_mse = np.array(epoch_test_mse).sum() /len(eval_loader)
    
    return avg_te_nll, avg_te_mse


def run_tr_te(args, 
              meta_net, 
              cat_dim, 
              net_optim, 
              train_loader, 
              eval_loader, 
              check_lvm, 
              whether_condition, 
              tr_loss_fun, 
              te_loss_fun, 
              tr_hard, 
              te_hard, 
              beta0, 
              beta1, 
              rand_eval, 
              writer):
    
    meta_tr_results, meta_te_nll_results, meta_te_mse_results = [], [], []
    for epoch in range(1, args.epochs + 1):
        avg_tr_nll = train_meta_net(epoch, meta_net, cat_dim, net_optim,
                                    train_loader, check_lvm, whether_condition, tr_loss_fun, tr_hard, beta0,
                                    beta1)
        meta_tr_results.append(avg_tr_nll)
        
        if rand_eval == True:
            avg_te_nll, avg_te_mse = eval_meta_net(epoch, meta_net, eval_loader,
                                                   check_lvm, whether_condition, te_loss_fun, te_hard, num_c_points=None)
            meta_te_nll_results.append(avg_te_nll)
            meta_te_mse_results.append(avg_te_mse)             
        
        else:    
            avg_te_nll1, avg_te_mse1 = eval_meta_net(epoch, meta_net, eval_loader,
                                                     check_lvm, whether_condition, te_loss_fun, te_hard, num_c_points=10)
            avg_te_nll2, avg_te_mse2 = eval_meta_net(epoch, meta_net, eval_loader,
                                                     check_lvm, whether_condition, te_loss_fun, te_hard, num_c_points=200)
            avg_te_nll3, avg_te_mse3 = eval_meta_net(epoch, meta_net, eval_loader,
                                                     check_lvm, whether_condition, te_loss_fun, te_hard, num_c_points=500)
            avg_te_nll4, avg_te_mse4 = eval_meta_net(epoch, meta_net, eval_loader,
                                                     check_lvm, whether_condition, te_loss_fun, te_hard, num_c_points=800)
            avg_te_nll5, avg_te_mse5 = eval_meta_net(epoch, meta_net, eval_loader,
                                                     check_lvm, whether_condition, te_loss_fun, te_hard, num_c_points=1000)
            
            meta_te_nll_results.append((avg_te_nll1,avg_te_nll2,avg_te_nll3,avg_te_nll4,avg_te_nll5))
            meta_te_mse_results.append((avg_te_mse1,avg_te_mse2,avg_te_mse3,avg_te_mse4,avg_te_mse5))            
        

        meta_tr_arr, meta_te_nll_arr, meta_te_mse_arr = np.array(meta_tr_results), \
            np.array(meta_te_nll_results), np.array(meta_te_mse_results)
        
        np.savetxt('./runs_results_cifar10/'+check_lvm+'/'+str(writer)+'/tr_loss_list.csv', meta_tr_arr)
        np.savetxt('./runs_results_cifar10/'+check_lvm+'/'+str(writer)+'/te_nll_list.csv', meta_te_nll_arr)
        np.savetxt('./runs_results_cifar10/'+check_lvm+'/'+str(writer)+'/te_mse_list.csv', meta_te_mse_arr)
        
        torch.save(meta_net.state_dict(),'./runs_results_cifar10/'+check_lvm+'/'+str(writer)+'/'+check_lvm+'.pth')
            
    
    return meta_tr_arr, meta_te_nll_arr, meta_te_mse_arr






