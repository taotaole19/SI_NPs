
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal 

from math import pi
from utils import *

    
        
        
class NP_VEM(nn.Module):
    '''
    Neural Process with variational expectation maximization methods.
    '''
    def __init__(self,args):
        super(NP_VEM,self).__init__()
        
        self.dim_x=args.dim_x
        self.dim_y=args.dim_y
        
        self.dim_h_lat=args.dim_h_lat 
        self.num_h_lat=args.num_h_lat 
        self.dim_lat=args.dim_lat
        self.num_particles=args.num_particles 
        
        self.dim_h=args.dim_h 
        self.num_h=args.num_h 
        self.act_type=args.act_type 
        self.nll_weight_vec=args.nll_weight_vec
        self.amort_y=args.amort_y  
        
        self.pred_enc_dist=args.pred_enc_dist 
        self.whether_baseline=args.whether_baseline
        self.prior_as_proposal=args.prior_as_proposal 
        
        self.prior_net=Context_Encoder(self.dim_x+self.dim_y, self.dim_h_lat, 
                                       self.act_type, self.num_h_lat, self.dim_lat).cuda()
        if not self.prior_as_proposal:
            self.proposal_net=Context_Encoder(self.dim_x+self.dim_y, self.dim_h_lat, 
                                              self.act_type, self.num_h_lat, self.dim_lat).cuda()
        
        self.dec_modules=[]
        self.dec_modules.append(nn.Linear(self.dim_x+self.dim_lat, self.dim_h))
        for i in range(self.num_h):
            self.dec_modules.append(get_act(self.act_type))
            self.dec_modules.append(nn.Linear(self.dim_h, self.dim_h))
        if self.amort_y:
            self.dec_modules.append(get_act(self.act_type))
            self.dec_modules.append(nn.Linear(self.dim_h, 2*self.dim_y)) 
        else:
            self.dec_modules.append(get_act(self.act_type))
            self.dec_modules.append(nn.Linear(self.dim_h, self.dim_y))         
        self.dec_net=nn.Sequential(*self.dec_modules).cuda() 

    
    def get_context_idx(self,M):
        N = random.randint(1,M)
        idx = random.sample(range(0, M), N)
        idx = torch.tensor(idx).cuda()
        
        return idx
        
        
    def idx_to_data(self,data,sample_dim,idx):
        ind_data= torch.index_select(data, dim=sample_dim, index=idx)
        
        return ind_data


    def prior_encoder(self,x_c,y_c):
        memo_c=torch.cat((x_c,y_c),dim=-1)
        mu, logvar = self.prior_net(memo_c)[0], self.prior_net(memo_c)[1]
        std=torch.exp(0.5*logvar)+1e-6*torch.ones_like(logvar).cuda()
        prior_dist=MultivariateNormal(mu,scale_tril=std.diag_embed()) 
        
        return prior_dist, mu, logvar
    
    
    def prior_loglike(self,prior_dist,batch_z):
        prior_logprob=torch.cat([(prior_dist.log_prob(batch_z[:,i,:])).unsqueeze(-1) for i in range(batch_z.size()[1])],dim=-1) 
        
        return prior_logprob
    
    
    def proposal_encoder(self,x_t,y_t):
        memo_t=torch.cat((x_t,y_t),dim=-1) 
        mu, logvar = self.proposal_net(memo_t)[0], self.proposal_net(memo_t)[1]
        std=torch.exp(0.5*logvar)
        proposal_dist=MultivariateNormal(mu,scale_tril=std.diag_embed()) 
        
        return proposal_dist, mu, logvar
    
    
    def proposal_loglike(self,proposal_dist,batch_z):
        proposal_logprob=torch.cat([(proposal_dist.log_prob(batch_z[:,i,:])).unsqueeze(-1) for i in range(batch_z.size()[1])],dim=-1) 
        
        return proposal_logprob 
    
    
    def proposal_sampler(self,proposal_dist,num_particles):
        sampled_batch_z=proposal_dist.rsample((num_particles,)) 
        batch_z=sampled_batch_z.permute((1,0,2)) 
        
        return batch_z      
            
    
    def cond_evidence_loglike(self,batch_z,x_t,y_t):
        batch_z_exp = batch_z.unsqueeze(1).expand(-1,x_t.size()[1],-1,-1) 
        x_t_exp = x_t.unsqueeze(2).expand(-1,-1,self.num_particles,-1)  
        output = self.dec_net(torch.cat((x_t_exp,batch_z_exp),dim=-1)) 
        
        assert self.amort_y == True
        mu_tensor, sigma_tensor = output[...,:self.dim_y],0.1+0.9*F.softplus(output[...,self.dim_y:]) 
        
        decoder_dist=MultivariateNormal(mu_tensor,scale_tril=sigma_tensor.diag_embed()) 
        
        y_t_exp=y_t.unsqueeze(2).expand(-1,-1,self.num_particles,-1)
        log_y_t=decoder_dist.log_prob(y_t_exp) 
        
        return decoder_dist, log_y_t
    
    
    def compute_normalized_iw(self,batch_z,prior_dist,proposal_dist,
                              log_y_t):
        log_cond_evidence=log_y_t.sum(1) 
        
        if self.prior_as_proposal:
            log_iw=log_cond_evidence 
            
        else:
            log_prior=self.prior_loglike(prior_dist,batch_z) 
            log_proposal=self.proposal_loglike(proposal_dist,batch_z) 
            log_iw=log_cond_evidence+log_prior-log_proposal 
          
        log_sum_iw=torch.logsumexp(log_iw, dim=-1).unsqueeze(-1) 

        norm_iw=torch.exp(log_iw-log_sum_iw)
        
        if self.prior_as_proposal:
            iw_baseline=0.0 
        else:
            log_sum_prior_proposal_ratio=torch.logsumexp(log_prior-log_proposal,dim=-1).unsqueeze(-1).expand(-1,self.num_particles) 
            iw_baseline=torch.exp(log_prior-log_proposal-log_sum_prior_proposal_ratio) 

        return norm_iw, iw_baseline
    
    
    def compute_iw_proposal_nll(self,proposal_dist,batch_z,norm_iw,whether_baseline=True):
        proposal_logprob=self.proposal_loglike(proposal_dist,batch_z) 
        if whether_baseline:
            proposal_baseline=(1.0/self.num_particles)*torch.ones_like(norm_iw)
            iw_logprob=(norm_iw-proposal_baseline)*proposal_logprob
        else:
            iw_logprob=norm_iw*proposal_logprob
        iw_proposal_nll=-iw_logprob.sum(-1).mean() 
        
        return iw_proposal_nll
    

    def reparameterization(self,mu,logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        
        return mu+eps*std

    
    def forward(self,x_c,y_c,x_t,y_t,x_pred,y_pred):
        # stochastic forward pass with multiple particles
        if not self.prior_as_proposal:
            proposal_dist, proposal_mu, proposal_logvar = self.proposal_encoder(x_t, y_t)
        
        prior_dist, prior_mu, prior_logvar = self.prior_encoder(x_c, y_c)
        
        if not self.prior_as_proposal:
            batch_z=self.proposal_sampler(proposal_dist,self.num_particles) 
        else:
            batch_z=self.proposal_sampler(prior_dist,self.num_particles)             
        
        decoder_dist,log_y_pred=self.cond_evidence_loglike(batch_z, x_pred, y_pred)
        
        prior_logprob=self.prior_loglike(prior_dist,batch_z) 
        
        if not self.prior_as_proposal:
            norm_iw, iw_baseline=self.compute_normalized_iw(batch_z,
                                                            prior_dist,
                                                            proposal_dist,
                                                            log_y_pred) 
        else:
            norm_iw, iw_baseline=self.compute_normalized_iw(batch_z,
                                                            prior_dist,
                                                            prior_dist,
                                                            log_y_pred)         
            
        norm_iw_d=norm_iw.detach()
        
        if not self.prior_as_proposal:
            iw_baseline_d=iw_baseline.detach()
        else:
            iw_baseline_d=iw_baseline
    
        if self.whether_baseline:
            iw_logprob=self.nll_weight_vec[0]*norm_iw_d*log_y_pred.sum(1)\
                +self.nll_weight_vec[1]*(norm_iw_d-iw_baseline)*prior_logprob 
        else:
            iw_logprob=self.nll_weight_vec[0]*norm_iw_d*log_y_pred.sum(1) \
                +self.nll_weight_vec[1]*norm_iw_d*prior_logprob 
        
        iw_nll=-iw_logprob.sum(-1).mean() 
        iw_logprob_evidence=norm_iw_d*log_y_pred.mean(1) 
        iw_nll_evidence=-iw_logprob_evidence.sum(-1).mean()
        
        if not self.prior_as_proposal:
            batch_z_d=batch_z.detach()
    
            iw_proposal_nll=self.nll_weight_vec[2]*self.compute_iw_proposal_nll(proposal_dist, batch_z_d, norm_iw_d)
        else:
            iw_proposal_nll=0.0
    
        return iw_nll, iw_proposal_nll, iw_nll_evidence
    
    
    def variance_value(self,x_c,y_c):
        if self.pred_enc_dist == 0:
            test_enc_dist, mu, logvar = self.prior_encoder(x_c,y_c)
        elif self.pred_enc_dist == 1:
            test_enc_dist, mu, logvar = self.proposal_encoder(x_c,y_c)
        
        covar_mat = torch.exp(logvar) 
        sum_var = covar_mat.sum(-1).mean() 
        
        return sum_var
      
    
    def conditional_predict(self,x_c,y_c,x_pred,y_pred,num_samples=32):
        # this method is for meta testing with multiple particles
        with torch.no_grad():   
            if self.pred_enc_dist == 0:
                test_enc_dist, mu, logvar = self.prior_encoder(x_c,y_c)
            elif self.pred_enc_dist == 1:
                test_enc_dist, mu, logvar = self.proposal_encoder(x_c,y_c)
            
            if num_samples==1:
                sampled_z = self.reparameterization(mu, logvar)
                z_exp = sampled_z.unsqueeze(1).expand(-1,x_pred.size()[1],-1)    
                output = self.dec_net(torch.cat((x_pred,z_exp),dim=-1)) 
                
                assert self.amort_y == True
                mu_tensor, sigma_tensor = output[...,:self.dim_y],0.1+0.9*F.softplus(output[...,self.dim_y:]) 
                
                decoder_dist=MultivariateNormal(mu_tensor,scale_tril=sigma_tensor.diag_embed()) 
                
                log_y_pred=decoder_dist.log_prob(y_pred) 
                b_avg_nll=-log_y_pred.mean()
                y_mean=mu_tensor
                
            else:
                sampled_batch_z=test_enc_dist.rsample((num_samples,))
                batch_z=sampled_batch_z.permute((1,0,2)) 
                batch_z_exp = batch_z.unsqueeze(1).expand(-1,x_pred.size()[1],-1,-1) 
                x_pred_exp = x_pred.unsqueeze(2).expand(-1,-1,num_samples,-1) 
                output = self.dec_net(torch.cat((x_pred_exp,batch_z_exp),dim=-1)) 
                
                assert self.amort_y == True
                mu_tensor, sigma_tensor = output[...,:self.dim_y],0.1+0.9*F.softplus(output[...,self.dim_y:]) 
                                    
                decoder_dist=MultivariateNormal(mu_tensor,scale_tril=sigma_tensor.diag_embed()) 
                
                y_pred_exp=y_pred.unsqueeze(2).expand(-1,-1,num_samples,-1) 
                log_y_pred=decoder_dist.log_prob(y_pred_exp) 
                sum_log_y_pred=log_y_pred.sum(-2)
                
                iw_logprob=(1.0/x_pred.size()[1])*(torch.logsumexp(sum_log_y_pred, dim=-1).mean())-(1.0/x_pred.size()[1])*torch.as_tensor(np.log(num_samples)).cuda()
                b_avg_nll=-iw_logprob
                y_mean=mu_tensor.mean(2)
                
            return mu, logvar, b_avg_nll, y_mean
        
        
    def complete(self, x_c, y_c, x_pred,y_pred):
        # this method is to generate new images as realizations
        with torch.no_grad():   
            if self.pred_enc_dist == 0:
                test_enc_dist, mu, logvar = self.prior_encoder(x_c,y_c)
            elif self.pred_enc_dist == 1:
                test_enc_dist, mu, logvar = self.proposal_encoder(x_c,y_c)
            
            sampled_z = self.reparameterization(mu, logvar)
            z_exp = sampled_z.unsqueeze(1).expand(-1,x_pred.size()[1],-1)    
            output = self.dec_net(torch.cat((x_pred,z_exp),dim=-1)) 
            
            assert self.amort_y == True
            mu_tensor, sigma_tensor = output[...,:self.dim_y],0.1+0.9*F.softplus(output[...,self.dim_y:]) 
            
            return mu_tensor, sigma_tensor             
            
                
    
