"""
Adopted from https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch
"""
from utils import *
import os
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import torch
import random
from PIL import Image
from .autoaugment import CIFAR10Policy, Cutout


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10
    dataset_name = 'CIFAR-10-LT'

    def __init__(self, phase, imbalance_ratio, root, imb_type='exp', cifar100_split='fine', autoaug=True):
        train = True if phase == "train" else False
        super(IMBALANCECIFAR10, self).__init__(root, train, transform=None, target_transform=None, download=True)
        
        if cifar100_split=='fine':
            pass
        elif cifar100_split=='coarse':
            print('Using coarse-grained CIFAR100 with 20 superclasses.')
            self.dataset_name = 'CIFAR-100-LT-Coarse'
            self.cls_num = 20
            meta = {
                'filename': 'meta',
                'key': 'coarse_label_names',
                'md5': '7973b15100ade9c7d40fb424638fde48',
            }
            ########        reload data with coarse labels          ########
            self.data: Any = []
            self.targets = []

            if self.train:
                downloaded_list = self.train_list
            else:
                downloaded_list = self.test_list

            # now load the picked numpy arrays
            for file_name, checksum in downloaded_list:
                file_path = os.path.join(self.root, self.base_folder, file_name)
                with open(file_path, "rb") as f:
                    entry = pickle.load(f, encoding="latin1")
                    self.data.append(entry["data"])
                    self.targets.extend(entry["coarse_labels"])

            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

            self._load_meta()
        else:
            raise Exception('Invalid cifar100 type. Either coarse or fine.')

        self.train = train

        if self.train:
            self.img_num_per_cls = self.get_img_num_per_cls(self.cls_num, imb_type, imbalance_ratio)
            self.gen_imbalanced_data(self.img_num_per_cls)
        
        self.autoaug = autoaug 
        if self.train and self.autoaug:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),    # add AutoAug
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif self.train:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        self.labels = self.targets
        if type(self.data) == torch.Tensor:
            self.data = self.data.data.cpu().numpy()

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        gamma = 1. / imb_factor
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (gamma ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * gamma))
        elif imb_type =='natural':
            for cls_idx in range(cls_num):
                img_num_per_cls.append(int((np.array(self.targets) == cls_idx).sum()))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        
        # save the class frequency
        if not os.path.exists('cls_freq'):
            os.makedirs('cls_freq')
        freq_path = os.path.join('cls_freq', self.dataset_name + '_IMBA{}.json'.format(imb_factor))
        with open(freq_path, 'w') as fd:
            json.dump(img_num_per_cls, fd)

        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls, seed=0):
        #################       deterministic training data         #################
        #############################################################################
        seed_everything(seed)       
        #############################################################################
        #############################################################################
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.labels)

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        for label in self.labels:
            annos.append({'category_id': int(label)})
        return annos

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    cls_num = 100                     
    dataset_name = 'CIFAR-100-LT'
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
