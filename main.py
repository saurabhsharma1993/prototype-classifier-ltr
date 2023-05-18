import numpy as np
import pickle
import os
import argparse
import pprint
from data import dataloader
from run_networks import model
import warnings
import yaml
from utils import *
import warnings
from data.ClassAwareSampler import ClassAwareSampler
warnings.filterwarnings("ignore")

data_root = {'ImageNet': './dataset/ImageNet_LT',
             'iNaturalist18': './dataset/iNaturalist18',
             'CIFAR100': './dataset/CIFAR100',
             }

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CIFAR100_LT', type=str)
parser.add_argument('--num_classes', default=100, type=int)
parser.add_argument('--cifar_imb_ratio', default=50, type=int)
parser.add_argument('--sampler', default='balanced', type=str)
parser.add_argument('--num_samples_cls', default=4, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--feat_dim', default=64, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=0.005, type=float)
parser.add_argument('--scheduler', default=None, type=str)
parser.add_argument('--centroids_lr', default=4.2, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=2e-4, type=float)
parser.add_argument('--num_epochs', default=1, type=int)
parser.add_argument('--display_step', default=40, type=int)
parser.add_argument('--logit_adj', default=3.8, type=float)    
parser.add_argument('--use_logit_adj', default=False, action='store_true')          
parser.add_argument('--pretrained_model_dir', default='./logs/CIFAR100_LT/wd_baseline_imba100', type=str)
parser.add_argument('--log_dir', default='./logs/CIFAR100_LT/prototype_classifier_imba100', type=str,)
parser.add_argument('--save_feat', default='', type=str)
parser.add_argument('--save', default=False, action='store_true')
parser.add_argument('--resnet34', default=False, action='store_true')
parser.add_argument('--autoaug', default=False, action='store_true')
parser.add_argument('--empirical_ncm', default=False, action='store_true')
parser.add_argument('--ncm_classifier', default=False, action='store_true')
parser.add_argument('--freeze_classifier', default=False, action='store_true')
parser.add_argument('--freeze_feats', default=False, action='store_true')
parser.add_argument('--overwrite', default=False, action='store_true')          # overwrite previous experiment
parser.add_argument('--test', default=False, action='store_true')

args = parser.parse_args()

test_mode = args.test
dataset = args.dataset
if args.autoaug:
    print('Using Autoaug and Cutout for train time data augmentation.')

if args.exp_num is not None:
    args.log_dir = os.path.join(args.log_dir, args.log_dir.split('/')[-1] + "_exp{}".format(args.exp_num))

args.feat_dim = 512

if not os.path.isdir(args.log_dir):
    os.makedirs(args.log_dir)

print('Loading dataset from: %s' % data_root[dataset.rstrip('_LT')])

if not test_mode:

    if args.sampler == 'balanced':
        sampler_dic = {
            'sampler': ClassAwareSampler,
            'params': {'num_samples_cls': args.num_samples_cls}
        }
    else:
        sampler_dic = None

    splits = ['train', 'val']
    if dataset not in ['iNaturalist18', 'ImageNet']:
        splits.append('test')
    data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                    dataset=dataset, phase=x, 
                                    batch_size=args.batch_size,
                                    sampler_dic=sampler_dic,
                                    num_workers=args.num_workers,
                                    autoaug=args.autoaug if x =='train' else False,
                                    cifar_imb_ratio=args.cifar_imb_ratio,
                                    cifar100_split='fine' if args.num_classes == 100 else 'coarse')
            for x in splits}

    data['train_save'] = dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                    dataset=dataset, phase='train', 
                                    batch_size=args.batch_size,
                                    sampler_dic=sampler_dic, 
                                    shuffle = False, 
                                    autoaug = False,
                                    num_workers=args.num_workers,
                                    cifar_imb_ratio=args.cifar_imb_ratio,
                                    cifar100_split='fine' if args.num_classes == 100 else 'coarse')

    if args.ncm_classifier:
        data['compute_centroids'] = dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                    dataset=dataset, phase='train',
                                    batch_size=args.batch_size,
                                    sampler_dic=None,   
                                    num_workers=args.num_workers,
                                    shuffle=False, 
                                    autoaug=True,        
                                    cifar_imb_ratio=args.cifar_imb_ratio,
                                    cifar100_split='fine' if args.num_classes == 100 else 'coarse',)

    training_model = model(args, data, test=False)
    training_model.train()
else:

    if 'iNaturalist' in args.dataset:
        splits = ['train', 'val']
        test_split = 'val'
    else:
        splits = ['train', 'val', 'test']
        test_split = 'test'
    if 'ImageNet' == args.dataset:
        splits = ['train', 'val']
        test_split = 'val'

    sampler_dic, shuffle = None, False

    data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                    dataset=dataset, phase=x,
                                    batch_size=args.batch_size,
                                    sampler_dic=None,    
                                    num_workers=args.num_workers,
                                    shuffle=False,
                                    autoaug=False,            
                                    cifar_imb_ratio=args.cifar_imb_ratio,
                                    cifar100_split='fine' if args.num_classes == 100 else 'coarse')
            for x in splits}

    if args.ncm_classifier:
        data['compute_centroids'] = dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                    dataset=dataset, phase='train',
                                    batch_size=args.batch_size,
                                    sampler_dic=None,   
                                    num_workers=args.num_workers,
                                    shuffle=False, 
                                    autoaug=False, 
                                    cifar_imb_ratio=args.cifar_imb_ratio,
                                    cifar100_split='fine' if args.num_classes == 100 else 'coarse',)
    
    training_model = model(args, data, test=True)       
    # model gets reloaded. first time is during init.
    training_model.load_model(args.log_dir)                    
    
    if args.save_feat in ['train', 'val', 'test']:
        saveit = True
        test_split = args.save_feat
    else:
        saveit = False
    
    training_model.eval(phase=test_split, save_feat=saveit)