
import argparse
import random
import torch

epochs=200
use_baseline=True
prior_as_proposal=True


def params_npvem():
    
    parser_npvem = argparse.ArgumentParser(description='npvem')
    parser_npvem.add_argument('--no-cuda', action='store_true', default=False,
                              help='disables CUDA training')
    parser_npvem.add_argument('--whether_baseline', default=use_baseline,
                              help='whether to use baseline in prior likelihood') 
    parser_npvem.add_argument('--prior_as_proposal', default=prior_as_proposal,
                              help='whether to use prior as the proposal dist')    
    parser_npvem.add_argument('--seed', type=int, default=1, metavar='S',
                              help='random seed (default: 1)')
    parser_npvem.add_argument('--log-interval', type=int, default=1, metavar='N',
                              help='how many batches to wait before logging training status')
    parser_npvem.add_argument('--batch_size', type=int, default=4, metavar='N',
                              help='input batch size for training')
    parser_npvem.add_argument('--epochs', type=int, default=epochs, metavar='N',
                              help='number of epochs to train')
    
    parser_npvem.add_argument('--dim_x', type=int, default=2, metavar='N',
                              help='dimension of input')
    parser_npvem.add_argument('--dim_y', type=int, default=3, metavar='N',
                              help='dimension of output')
    
    parser_npvem.add_argument('--dim_h_lat', type=int, default=128, metavar='N',
                              help='dim of hidden units for encoders')
    parser_npvem.add_argument('--num_h_lat', type=int, default=3, metavar='N',
                              help='num of layers for encoders')
    parser_npvem.add_argument('--dim_lat', type=int, default=128, metavar='N',
                              help='dimension of z, the global latent variable') 
    parser_npvem.add_argument('--num_particles', type=int, default=8, metavar='N',
                              help='number of sampled latent variables for normalized IS')    
    
    
    parser_npvem.add_argument('--dim_h', type=int, default=128, metavar='N',
                              help='dim of hidden units for decoders')   
    parser_npvem.add_argument('--num_h', type=int, default=5, metavar='N',
                              help='num of layers for decoders') 
    parser_npvem.add_argument('--act_type', type=str, default='ReLU', metavar='N',
                              help='type of activation units')   
    parser_npvem.add_argument('--amort_y', type=bool, default=True, metavar='N',
                              help='whether to amortize output distributions')
     
    parser_npvem.add_argument('--nll_weight_vec', default=[1.0, 1.0, 0.0], metavar='N',
                               help='weight vector for nll terms')    
    parser_npvem.add_argument('--pred_enc_dist', default=0, metavar='N',
                              help='use prior or proposal in test')    
    
    args = parser_npvem.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    return args,random,device


