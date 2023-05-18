import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
import torch.nn.functional as F
import importlib
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler
import pdb
import math
import random
import os
import pickle

class CosineAnnealingLRWarmup(_LRScheduler):
	"""
	Cosine Annealing with Warm Up.
	"""
	def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, warmup_epochs=5, base_lr=0.05, warmup_lr=0.1):
		self.T_max = T_max
		self.eta_min = eta_min
		self.warmup_epochs = warmup_epochs
		self.base_lr = base_lr
		self.warmup_lr = warmup_lr
		super(CosineAnnealingLRWarmup, self).__init__(optimizer, last_epoch)

	def get_cos_lr(self):
		return [self.eta_min + (self.warmup_lr - self.eta_min) *
				(1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs))) / 2
				/ self.base_lr * base_lr
				for base_lr in self.base_lrs]

	def get_warmup_lr(self):
		return [((self.warmup_lr - self.base_lr) / (self.warmup_epochs-1) * (self.last_epoch - 1)
				+ self.base_lr) / self.base_lr * base_lr
				for base_lr in self.base_lrs]

	def get_lr(self):
		assert self.warmup_epochs >= 2
		if self.last_epoch < self.warmup_epochs:
			return self.get_warmup_lr()
		else:
			return self.get_cos_lr()

def batch_show(inp, title=None):
	"""Imshow for Tensor."""
	inp = inp.numpy().transpose((1, 2, 0))
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	inp = std * inp + mean
	inp = np.clip(inp, 0, 1)
	plt.figure(figsize=(20,20))
	plt.imshow(inp)
	if title is not None:
		plt.title(title)

def print_write(print_str, log_file):
	print(*print_str)
	if log_file is None:
		return
	with open(log_file, 'a') as f:
		print(*print_str, file=f)

def shot_acc (preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
	if isinstance(train_data, np.ndarray):
		training_labels = np.array(train_data).astype(int)
	else:
		training_labels = np.array(train_data.dataset.labels).astype(int)
	if isinstance(preds, torch.Tensor):
		preds = preds.detach().cpu().numpy()
		labels = labels.detach().cpu().numpy()
	elif isinstance(preds, np.ndarray):
		pass
	else:
		raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
	train_class_count = []
	test_class_count = []
	class_correct = []
	for l in np.unique(labels):
		train_class_count.append(len(training_labels[training_labels == l]))
		test_class_count.append(len(labels[labels == l]))
		class_correct.append((preds[labels == l] == labels[labels == l]).sum())
	many_shot = []
	median_shot = []
	low_shot = []
	for i in range(len(train_class_count)):
		if train_class_count[i] > many_shot_thr:
			many_shot.append((class_correct[i] / test_class_count[i]))
		elif train_class_count[i] < low_shot_thr:
			low_shot.append((class_correct[i] / test_class_count[i]))
		else:
			median_shot.append((class_correct[i] / test_class_count[i]))    
	if len(many_shot) == 0:
		many_shot.append(0)
	if len(median_shot) == 0:
		median_shot.append(0)
	if len(low_shot) == 0:
		low_shot.append(0)
	if acc_per_cls:
		class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)] 
		return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
	else:
		return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)

def mic_acc_cal(preds, labels):
	if isinstance(labels, tuple):
		assert len(labels) == 3
		targets_a, targets_b, lam = labels
		acc_mic_top1 = (lam * preds.eq(targets_a.data).cpu().sum().float() \
					   + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()) / len(preds)
	else:
		acc_mic_top1 = (preds == labels).sum().item() / len(labels)
	return acc_mic_top1

def next_balanced_batch(out, y, num_cls, num_per_class, device):
	bal_out = torch.empty(num_cls*num_per_class,out.shape[1]).to(device)
	bal_y   = torch.empty(num_cls*num_per_class).long().to(device)
	for cls in range(num_cls):
		cls_idx = np.where(y.cpu() == cls)[0]
		samp_idx = np.random.choice(cls_idx.shape[0],num_per_class)
		cls_samp_idx = cls_idx[samp_idx]
		bal_out[cls*num_per_class:(cls+1)*num_per_class] = out[cls_samp_idx,:]
		bal_y[cls*num_per_class:(cls+1)*num_per_class] = y[cls_samp_idx]
	return bal_out, bal_y

def torch2numpy(x):
	if isinstance(x, torch.Tensor):
		return x.detach().cpu().numpy()
	elif isinstance(x, (list, tuple)):
		return tuple([torch2numpy(xi) for xi in x])
	else:
		return x

def seed_everything(seed=1234):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	torch.backends.cudnn.deterministic = True