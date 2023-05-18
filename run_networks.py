import numpy as np
import os
import copy
import pprint
import pickle
from scipy.linalg import pinvh
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
import time
import warnings
import pdb
import higher
from utils import *
from models.ResNet32Feature import BasicBlock, ResNet_Cifar, ResnetEncoder 
from models.ResNextFeature import ResNext
from models.ResNextFeature import Bottleneck as ResNeXtBottleneck
from models.ResNet50Feature import ResNet
from models.ResNet50Feature import Bottleneck as ResNetBottleneck
from models.DotProductClassifier import DotProduct_Classifier
from models.NcmClassifier import NcmClassifier
import torchvision.transforms as transforms

class model ():
    
    def __init__(self, args, data, test=False):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.data = data
        self.test_mode = test
        self.num_gpus = torch.cuda.device_count()
        self.num_classes = self.args.num_classes
        self.clssizes = self.compute_classsizes()

        # Set up log file
        self.log_file = os.path.join(self.args.log_dir, 'log.txt')
        if not self.test_mode and os.path.isfile(self.log_file):
            if self.args.overwrite:
                os.remove(self.log_file)
                print('Overwriting experiment log file, models etc.')
            else:
                raise Exception('Experiment log file already exists! Are you sure?')
                    
        # init model parameters
        self.centroids, self.stddevs = None, None
        self.ncm_classifier = self.args.ncm_classifier
        
        # Initialize model
        self.init_models()    
        # Load pre-trained model parameters
        if self.args.pretrained_model_dir is not None:
            self.load_model(self.args.pretrained_model_dir)

        self.cls_num_list = torch.Tensor(self.data['train'].dataset.get_cls_num_list()).cuda()
        self.criterion = nn.CrossEntropyLoss()

        self.logit_adj = torch.scalar_tensor(args.logit_adj).cuda()

        # Under training mode, initialize training steps, optimizers, schedulers, criterions, and centroids
        if not self.test_mode:
            self.training_data_num = len(self.data['train'].dataset)
            self.epoch_steps = int(self.training_data_num / self.args.batch_size) # for recomputing center
            # Initialize model optimizer and scheduler
            self.model_optimizer = optim.SGD(self.model_optim_params_list)       
            if args.scheduler == 'cos':
                self.model_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.model_optimizer, self.args.num_epochs, eta_min=0)     
            for arg in vars(self.args):
                print_write([arg, getattr(self.args, arg)], self.log_file)
        else:
            self.log_file = None    

    def init_models(self, optimizer=True):
        if self.ncm_classifier:
            self.classifier = NcmClassifier(self.num_classes, self.args.feat_dim).cuda()
        else:
            self.classifier = DotProduct_Classifier(self.num_classes, 512).cuda()                            
        if self.args.dataset == 'ImageNet_LT':
            self.feature_extractor = nn.DataParallel(ResNext(ResNeXtBottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, num_classes = self.args.num_classes).cuda())
        elif self.args.dataset == 'iNaturalist18':
            self.feature_extractor = nn.DataParallel(ResNet(ResNetBottleneck, [3, 4, 6, 3], num_classes = self.args.num_classes).cuda())
        else:
            self.feature_extractor = ResnetEncoder(34, False, embDimension=self.num_classes, poolSize=4).cuda()  
        if self.args.freeze_classifier:
            for param_name, param in self.classifier.named_parameters():
                param.requires_grad = False
        if self.args.freeze_feats:
            for param_name, param in self.feature_extractor.named_parameters():
                param.requires_grad = False
        self.model_optim_params_list = []
        self.model_optim_params_list.append({'params': self.feature_extractor.parameters(), 'lr': self.args.lr, 'momentum': self.args.momentum, 'weight_decay': self.args.weight_decay})
        self.model_optim_params_list.append({'params': self.classifier.parameters(), 'lr': self.args.lr, 'momentum': self.args.momentum, 'weight_decay': self.args.weight_decay})

    def compute_classsizes(self):
    
        total_labels = np.array(self.data['train'].dataset.labels)

        clssizes = []
        for cls in range(self.num_classes):
            clsidx = total_labels==cls
            clssize = clsidx.sum()
            clssizes.append(clssize)

        return clssizes

    def batch_forward(self, inputs, labels=None, phase='train'):

        # Calculate Features
        self.features, self.feature_maps = self.feature_extractor(inputs)        
        # # Skip logits if computing centroids
        if phase == 'compute_centroids':
            return labels                
        # Calculate logits with classifier
        if self.ncm_classifier:
            self.logits, _ = self.classifier(self.features, self.centroids, self.stddevs, phase)
        else:
            self.logits, _ = self.classifier(self.features)
        if self.args.use_logit_adj and phase=='train':
            self.logits = self.logits + 1/self.logit_adj*torch.log(self.cls_num_list[None,:].float().squeeze()) 
        return labels

    def batch_backward(self):
        # Zero out optimizer gradients
        self.model_optimizer.zero_grad()
        # Back-propagation from loss outputs
        self.loss.backward(retain_graph = True)
        # (optional) gradient clipping
        # torch.nn.utils.clip_grad_value_(self.centroids,0.1)
        # Step optimizers
        self.model_optimizer.step()
        
    def batch_loss(self, labels):
        self.loss = 0 
        self.loss_perf = self.criterion(self.logits,labels)
        self.loss += self.loss_perf

    def train(self):
        # When training the network
        print_str = ['Phase: train']
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        # Initialize best model
        best_model_weights = {}
        best_model_weights['feature_extractor'] = copy.deepcopy(self.feature_extractor.state_dict())
        best_model_weights['classifier'] = copy.deepcopy(self.classifier.state_dict())
        best_acc, best_epoch, best_centroids = 0.0, 0, self.centroids
        end_epoch = self.args.num_epochs
        lr = self.args.lr if not self.args.ncm_classifier else self.args.centroids_lr

        #################       epoch loop                #################
        for epoch in range(1, end_epoch + 1):
            
            self.epoch = epoch 
            self.feature_extractor.train()
            self.classifier.train()          

            if epoch == 1 and self.ncm_classifier:
                if self.centroids is None:      
                    self.eval(phase='compute_centroids', printit=False) 
                if self.args.empirical_ncm:
                    self.centroids.requires_grad = False
                else:
                    self.centroids.requires_grad = True
                    centroids_lr = self.args.centroids_lr
                    self.model_optim_params_list.append({'params': self.centroids,'lr': centroids_lr, 'momentum': 0.9})
                    self.model_optimizer = optim.SGD(self.model_optim_params_list)
                    self.model_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.model_optimizer, self.args.num_epochs, eta_min=0)     

            torch.cuda.empty_cache()

            if self.args.scheduler == 'cos':
                self.model_scheduler.step() 
                lr = self.model_scheduler.get_last_lr()[-1]         
            elif not self.args.freeze_feats:
                lr = self.adjust_learning_rate(epoch)               
           
            # Iterate over dataset
            total_preds = []
            total_labels = []
            total_features = []

            ###############         mini-batch loop               ###############
            for step, (inputs, labels, indexes) in enumerate(self.data['train']):

                inputs, labels = inputs.cuda(), labels.cuda()

                # If on training phase, enable gradients
                with torch.set_grad_enabled(True):
                    
                    self.batch_forward(inputs, labels, 
                                       phase='train')
                    self.batch_loss(labels)
                    if not self.args.empirical_ncm:
                        self.batch_backward()               

                    # Tracking predictions
                    _, preds = torch.max(self.logits, 1)
                    total_preds.append(torch2numpy(preds))
                    total_labels.append(torch2numpy(labels))
                    total_features.append(torch2numpy(self.features))

                    ## Output minibatch training results
                    if step % self.args.display_step == 0:
                        minibatch_loss_perf, minibatch_loss_total, minibatch_acc = self.loss_perf.item(), self.loss.item(), mic_acc_cal(preds, labels)
                        minibatch_loss_feat = None
                        print_str = ['Epoch: [%d/%d]' % (epoch, self.args.num_epochs), 'Step: %5d' % (step), 'LR: %.3f' % lr, 'Softmax: %.3f' % (minibatch_loss_perf), 'Accuracy: %.3f' % (minibatch_acc)]
                        print_write(print_str, self.log_file)

            # After every epoch, validation. 
            self.eval_with_preds(total_preds, total_labels,printit=False)
            if self.args.save:
                self.eval(phase='train_save',printit=True, save_feat=True, epoch=epoch)
            self.eval(phase='val',printit=True)             
            self.total_features, self.total_preds, self.total_labels = total_features, total_preds, total_labels

            # Under validation, the best model need to be updated
            if self.eval_acc_mic_top1 > best_acc:
                best_epoch, best_acc, best_centroids = epoch, self.eval_acc_mic_top1, self.centroids.detach().clone() if self.centroids is not None else None
                best_model_weights['feature_extractor'] = copy.deepcopy(self.feature_extractor.state_dict())
                best_model_weights['classifier'] = copy.deepcopy(self.classifier.state_dict())
            
            print('===> Saving checkpoint')
            self.save_latest(epoch)

            if self.args.empirical_ncm:
                # run for just 1 epoch
                break

        print()
        print_write(['Training Complete.'], self.log_file)

        print_str = ['Best validation accuracy is %.3f at epoch %d' % (best_acc, best_epoch)]
        print_write(print_str, self.log_file)
        # Save the best model and best centroids if calculated
        self.save_model(epoch, best_epoch, best_model_weights, best_acc, centroids=best_centroids)
        # Test on the test set
        self.reset_model(best_model_weights, centroids = best_centroids)
        self.eval('test' if 'test' in self.data else 'val')
    
    def eval_with_preds(self, preds, labels, printit=True):
        n_total = sum([len(p) for p in preds])
        all_preds, all_labels = [], []
        for p, l in zip(preds, labels):
            all_preds.append(p)
            all_labels.append(l)
    
        # Calculate prediction accuracy
        rsl = {'train_all':0., 'train_many':0., 'train_median':0., 'train_low': 0.}
        if len(all_preds) > 0:
            all_preds, all_labels = list(map(np.concatenate, [all_preds, all_labels]))
            n_top1 = mic_acc_cal(all_preds, all_labels)
            n_top1_many, n_top1_median, n_top1_low, = shot_acc(all_preds, all_labels, self.data['train'])
            rsl['train_all'] += len(all_preds) / n_total * n_top1
            rsl['train_many'] += len(all_preds) / n_total * n_top1_many
            rsl['train_median'] += len(all_preds) / n_total * n_top1_median
            rsl['train_low'] += len(all_preds) / n_total * n_top1_low

        # Top-1 accuracy and additional string
        print_str = ['\n Training acc Top1: %.3f \n' % (rsl['train_all']), 'Many_top1: %.3f' % (rsl['train_many']), 'Median_top1: %.3f' % (rsl['train_median']), 'Low_top1: %.3f' % (rsl['train_low']), '\n']
        if printit:
            print_write(print_str, self.log_file)

        return rsl

    ############ evaluate and save features and other things ############
    def eval(self, phase='val', save_feat=False, printit=True, epoch=None):

        if printit:
            print_str = ['Phase: %s' % (phase)]
            print_write(print_str, self.log_file)
            time.sleep(0.25)
        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval() and initialize running loss/correct    
        self.feature_extractor.eval()
        self.classifier.eval()

        get_feat_only = save_feat
        feats_all, labels_all, idxs_all, logits_all = [], [], [], []
        featmaps_all = []
        # Iterate over dataset
        for inputs, labels, paths in tqdm(self.data[phase]):
            inputs, labels = inputs.cuda(), labels.cuda()

            with torch.set_grad_enabled(False):

                # In validation or testing
                self.batch_forward(inputs, labels,
                                   phase=phase)
                
                if not phase == 'compute_centroids':
                    noflip_logits = self.logits.detach().clone()
                    inputs = inputs.flip(3)
                    self.batch_forward(inputs, labels,
                                       phase=phase)
                    self.logits = (self.logits + noflip_logits)/2

                # Skip logits if computing centroids
                if not phase == 'compute_centroids':
                    logits_all.append(self.logits.cpu().numpy())
                feats_all.append(self.features.cpu().numpy())
                labels_all.append(labels.cpu().numpy())
                idxs_all.append(paths.numpy())

        self.total_features = np.concatenate(feats_all)
        self.total_labels = np.concatenate(labels_all)
        # Skip logits if computing centroids
        if not phase == 'compute_centroids':
            self.total_logits = np.concatenate(logits_all)   
        self.total_paths = np.concatenate(idxs_all)

        # recompute centroids
        if phase =='compute_centroids':
            self.compute_centroids()
            print_write(['Centroids computed.'], self.log_file)
            return

        if get_feat_only:
            name = '{}feat_all.pkl'.format(phase)
            if epoch != None:
                name = 'epoch{}_'.format(epoch) + name
            fname = os.path.join(self.args.log_dir, name)
            print('===> Saving feats to ' + fname)
            with open(fname, 'wb') as f:
                pickle.dump({'feats': self.total_features, 'labels': self.total_labels, 'logits': self.total_logits,'idxs': self.total_paths},f, protocol=4) 

        self.total_logits, self.total_labels = torch.from_numpy(self.total_logits).cuda(), torch.from_numpy(self.total_labels).cuda()
        probs, preds = F.softmax(self.total_logits.detach(), dim=1).max(dim=1)

        # Calculate the overall accuracy and F measurement
        self.eval_acc_mic_top1= mic_acc_cal(preds, self.total_labels)
        self.many_acc_top1, self.median_acc_top1, self.low_acc_top1, self.cls_accs = shot_acc(preds,self.total_labels, self.data['train'], acc_per_cls=True)
        # Top-1 accuracy and additional string
        print_str = ['Phase: %s' % (phase),'All: %.4f' % (self.eval_acc_mic_top1), 'Many: %.4f' % (self.many_acc_top1), 'Medium: %.4f' % (self.median_acc_top1), 'Few: %.4f' % (self.low_acc_top1)]
        rsl = {phase + '_all': self.eval_acc_mic_top1, phase + '_many': self.many_acc_top1, phase + '_median': self.median_acc_top1, phase + '_low': self.low_acc_top1,}
        cls_accs = {'cls_accs': self.cls_accs}

        with open(os.path.join(self.args.log_dir, 'cls_accs_{}.pkl'.format(phase)), 'wb') as f:
            pickle.dump(cls_accs, f)

        with open(os.path.join(self.args.log_dir, 'rsl_{}.pkl'.format(phase)), 'wb') as f:
            pickle.dump(rsl, f)

        if printit:
            if phase == 'val':
                print_write(print_str, self.log_file)
            else:
                acc_str = ["{:.2f} \t {:.2f} \t {:.2f} \t {:.2f}".format(self.many_acc_top1 * 100, self.median_acc_top1 * 100, self.low_acc_top1 * 100, self.eval_acc_mic_top1 * 100)]
                if self.log_file is not None and os.path.exists(self.log_file):
                    print_write(print_str, self.log_file)
                    print_write(acc_str, self.log_file)
                else:
                    print(*print_str)
                    print(*acc_str)
        
        return rsl 
    
    ############ compute centroids ############
    def compute_centroids(self):   
        centroids, stddevs, ranges = [], [], []
        epsilon = 1e-1       
        for cls in range(self.num_classes):          
            cls_idx = np.where(self.total_labels == cls)[0]
            cls_size = len(cls_idx)
            cls_feats = self.total_features[cls_idx]            
            cls_mean = np.mean(cls_feats,axis=0)
            centroids.append(cls_mean)           
            cls_stddev = np.mean(np.linalg.norm(cls_feats - cls_mean[None,:],axis=1)) 
            stddevs.append(cls_stddev)
            cls_range = np.max(np.linalg.norm(cls_feats[None,:,:]-cls_feats[:,None,:],axis=2))
            ranges.append(cls_range)
        self.centroids = torch.from_numpy(np.vstack(centroids)).cuda()
        self.stddevs = torch.from_numpy(np.array(stddevs)).cuda()
        self.ranges = torch.from_numpy(np.array(ranges)).cuda()

    def reset_model(self, model_state, centroids = None):
        networks = {'feature_extractor': self.feature_extractor, 'classifier': self.classifier}
        for key, model in networks.items():
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            model.load_state_dict(weights)
        if centroids is not None:
            self.centroids = centroids

    def load_model(self, model_dir=None):

        if not model_dir.endswith('.pth'):
            model_dir = os.path.join(model_dir, 'final_model_checkpoint.pth')
        
        if not self.test_mode:
            print_write(['Loading model from %s' % (model_dir)], self.log_file)
        else:
            print('Loading model from %s' % (model_dir))

        checkpoint = torch.load(model_dir)          
        model_state = checkpoint['state_dict_best']
        self.centroids = checkpoint['centroids'] if 'centroids' in checkpoint else None
        networks = {'feature_extractor': self.feature_extractor}
        if not self.args.ncm_classifier or self.args.test:
            networks['classifier'] = self.classifier
        for key, model in networks.items():
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            x = model.state_dict()
            x.update(weights)
            model.load_state_dict(x)
    
    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        epoch = epoch + 1
        if epoch <= 5:
            lr = self.args.lr * epoch / 5
        elif epoch > 180:
            lr = self.args.lr * 0.0001
        elif epoch > 160:
            lr = self.args.lr * 0.01
        else:
            lr = self.args.lr
        for param_group in self.model_optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def save_latest(self, epoch):
        model_weights = {}
        model_weights['feature_extractor'] = copy.deepcopy(self.feature_extractor.state_dict())
        model_weights['classifier'] = copy.deepcopy(self.classifier.state_dict())
        model_states = {'epoch': epoch, 'state_dict': model_weights}
        model_dir = os.path.join(self.args.log_dir, 'latest_model_checkpoint.pth')
        torch.save(model_states, model_dir)
        
    def save_model(self, epoch, best_epoch, best_model_weights, best_acc, centroids=None):
        model_states = {'epoch': epoch,'best_epoch': best_epoch, 'state_dict_best': best_model_weights, 'best_acc': best_acc, 'centroids': centroids,}
        model_dir = os.path.join(self.args.log_dir, 'final_model_checkpoint.pth')
        torch.save(model_states, model_dir)