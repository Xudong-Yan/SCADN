"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
import torch
import glob
import random
import numpy as np
import torchvision.datasets as datasets
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageFolder
from PIL import Image
import torchvision.transforms as transforms


class BlockMask(torch.utils.data.Dataset):
    def __init__(self, config):
        self.flist = list(glob.glob(config.TRAIN_MASK_FLIST + '/*.jpg')) + list(glob.glob(config.TRAIN_MASK_FLIST + '/*.png'))
        self.flist.sort()
        self.mask_set = []
        for mask_index in range(len(self.flist)):
            mask = Image.open(self.flist[mask_index])
            mask = transforms.Resize(config.INPUT_SIZE, Image.NEAREST)(mask)
            self.mask_set.append(transforms.ToTensor()(mask))

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, index):
        return self.mask_set[index]


def load_data(opt):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    masks = AddMask(opt)
    if opt.normal_class == '':
        opt.normal_class = opt.SUB_SET
    # LOAD DATA SET
    if opt.dataroot == '':
        opt.dataroot = './data/{}'.format(opt.DATASET)

    if opt.DATASET in ['cifar10']:
        splits = ['train', 'test', 'train4val']
        drop_last_batch = {'train': True, 'test': False, 'train4val': False}
        shuffle = {'train': True, 'test': False, 'train4val': False}

        transform_train = transforms.Compose(
            [
                transforms.Resize(opt.INPUT_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )# the train augment mentioned in paper
        transform_test = transforms.Compose(
            [
                transforms.Resize(opt.INPUT_SIZE),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        classes = {
            'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
            'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
        }

        dataset = {}
        dataset['train'] = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        dataset['test'] = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        dataset['train4val'] = CIFAR10(root='./data', train=True, download=True, transform=transform_test)
        dataset['test_copy'] = CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        dataset['train'].data, dataset['train'].targets, \
        dataset['test'].data, dataset['test'].targets = get_cifar_anomaly_dataset(
            trn_img=dataset['train'].data,
            trn_lbl=dataset['train'].targets,
            tst_img=dataset['test'].data,
            tst_lbl=dataset['test'].targets,
            nrm_cls_idx=classes[opt.normal_class],
            manualseed=opt.SEED
        )
        dataset['train4val'].data, dataset['train4val'].targets, _, _ = get_cifar_anomaly_dataset(
            trn_img=dataset['train4val'].data,
            trn_lbl=dataset['train4val'].targets,
            tst_img=dataset['test_copy'].data,
            tst_lbl=dataset['test_copy'].targets,
            nrm_cls_idx=classes[opt.normal_class],
            manualseed=opt.SEED
        )

        # collate = {'train': collate_ITAE, 'test': collate_ITAE_eval, 'train4val': collate_ITAE_eval}
        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.BATCH_SIZE,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     collate_fn=masks.append_mask,
                                                     worker_init_fn=(None if opt.SEED == -1
                                                     else lambda x: np.random.seed(opt.SEED)))
                      for x in splits}
        return dataloader

    elif opt.DATASET in ['mnist']:
        opt.normal_class = int(opt.normal_class)

        splits = ['train', 'test', 'train4val']
        drop_last_batch = {'train': True, 'test': False, 'train4val': False}
        shuffle = {'train': True, 'test': True, 'train4val': False}

        # no augment used on mnist dataset in the paper
        transform = transforms.Compose(
            [
                transforms.Resize(opt.INPUT_SIZE),
                transforms.ToTensor(),
                # transforms.Normalize((0.5,), (0.5,))
                # transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        dataset = {}
        dataset['train'] = MNIST(root='./data', train=True, download=True, transform=transform)
        dataset['test'] = MNIST(root='./data', train=False, download=True, transform=transform)
        dataset['train4val'] = MNIST(root='./data', train=True, download=True, transform=transform)
        dataset['test_copy'] = MNIST(root='./data', train=False, download=True, transform=transform)


        dataset['train'].data, dataset['train'].targets, \
        dataset['test'].data, dataset['test'].targets = get_mnist_anomaly_dataset(
            trn_img=dataset['train'].data,
            trn_lbl=dataset['train'].targets,
            tst_img=dataset['test'].data,
            tst_lbl=dataset['test'].targets,
            nrm_cls_idx=opt.normal_class,
            manualseed=opt.SEED
        )
        dataset['train4val'].data, dataset['train4val'].targets, _, _ = get_mnist_anomaly_dataset(
            trn_img=dataset['train4val'].data,
            trn_lbl=dataset['train4val'].targets,
            tst_img=dataset['test_copy'].data,
            tst_lbl=dataset['test_copy'].targets,
            nrm_cls_idx=opt.normal_class,
            manualseed=opt.SEED
        )

        # collate = {'train': collate_ITAE, 'test': collate_ITAE_eval, 'train4val': collate_ITAE_eval}
        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.BATCH_SIZE,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     collate_fn=masks.append_mask,
                                                     worker_init_fn=(None if opt.SEED == -1
                                                     else lambda x: np.random.seed(opt.SEED)))
                      for x in splits}
        return dataloader

    else:
        if opt.SUB_SET is not None:
            opt.dataroot = os.path.join(opt.dataroot, opt.SUB_SET)
        splits = ['train', 'test', 'train4val']
        splits2folder = {'train': 'train', 'test': 'test', 'train4val': 'train'}
        drop_last_batch = {'train': True, 'test': False, 'train4val': False}
        shuffle = {'train': True, 'test': False, 'train4val': False}
        transform = transforms.Compose([transforms.Resize(opt.INPUT_SIZE),
                                        transforms.CenterCrop(opt.INPUT_SIZE),
                                        transforms.ToTensor(), ])
                                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

        collate = {'train': masks.append_mask, 'test': masks.append_mask, 'train4val': masks.append_mask}
        dataset = {x: ImageFolder(os.path.join(opt.dataroot, splits2folder[x]), transform) for x in splits}


        dataset = {x: get_custom_anomaly_dataset(dataset[x], opt.normal_class) for x in dataset.keys()}
        # dataset['train4val'] = dataset['train']
        dataloader = {}
        for x in splits:
            if collate[x] is not None:
                dataloader[x] = torch.utils.data.DataLoader(dataset=dataset[x],
                                                            batch_size=opt.BATCH_SIZE,
                                                            shuffle=shuffle[x],
                                                            num_workers=int(opt.workers),
                                                            drop_last=drop_last_batch[x],
                                                            collate_fn=collate[x],
                                                            worker_init_fn=(None if opt.SEED == -1
                                                            else lambda x: np.random.seed(opt.SEED)))
            else:
                dataloader[x] = torch.utils.data.DataLoader(dataset=dataset[x],
                                                            batch_size=opt.BATCH_SIZE,
                                                            shuffle=shuffle[x],
                                                            num_workers=int(opt.workers),
                                                            drop_last=drop_last_batch[x],
                                                            worker_init_fn=(None if opt.SEED == -1
                                                                            else lambda x: np.random.seed(opt.SEED)))
        return dataloader

##
def get_cifar_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, nrm_cls_idx=0, manualseed=-1):
    """[summary]

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        nrm_cls_idx {int} -- normal class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.

    """
    # Convert train-test labels into numpy array.
    trn_lbl = np.array(trn_lbl)
    tst_lbl = np.array(tst_lbl)

    # --
    # Find idx, img, lbl for abnormal and normal on org dataset.
    nrm_trn_idx = np.where(trn_lbl == nrm_cls_idx)[0]
    abn_trn_idx = np.where(trn_lbl != nrm_cls_idx)[0]
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.

    nrm_tst_idx = np.where(tst_lbl == nrm_cls_idx)[0]
    abn_tst_idx = np.where(tst_lbl != nrm_cls_idx)[0]
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # Create new anomaly dataset based on the following data structure:
    # - anomaly dataset
    #   . -> train
    #        . -> normal
    #   . -> test
    #        . -> normal
    #        . -> abnormal
    new_trn_img = np.copy(nrm_trn_img)
    new_trn_lbl = np.copy(nrm_trn_lbl)
    # new_tst_img = np.concatenate((nrm_tst_img, abn_trn_img, abn_tst_img), axis=0)
    # new_tst_lbl = np.concatenate((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), axis=0)
    new_tst_img = np.concatenate((nrm_tst_img, abn_tst_img), axis=0)
    new_tst_lbl = np.concatenate((nrm_tst_lbl, abn_tst_lbl), axis=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl

##
def get_mnist_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, nrm_cls_idx=0, manualseed=-1):
    """[summary]

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        nrm_cls_idx {int} -- Normal class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.

    """
    # --
    # Find normal abnormal indexes.
    nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == nrm_cls_idx)[0])
    abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() != nrm_cls_idx)[0])
    nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() == nrm_cls_idx)[0])
    abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() != nrm_cls_idx)[0])

    # --
    # Find normal and abnormal images
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images.
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.

    # --
    # Find normal and abnormal labels.
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1


    # Create new anomaly dataset based on the following data structure:
    new_trn_img = nrm_trn_img.clone()
    new_trn_lbl = nrm_trn_lbl.clone()
    # new_tst_img = torch.cat((nrm_tst_img, abn_trn_img, abn_tst_img), dim=0)
    # new_tst_lbl = torch.cat((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), dim=0)
    new_tst_img = torch.cat((nrm_tst_img, abn_tst_img), dim=0)
    new_tst_lbl = torch.cat((nrm_tst_lbl, abn_tst_lbl), dim=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl


def get_custom_anomaly_dataset(subset, nrm_cls):
    nrm_cls_idx = subset.class_to_idx[nrm_cls]
    idx_to_class = {v: k for k, v in subset.class_to_idx.items()}
    new_targets = [0 if x == nrm_cls_idx else 1 for x in subset.targets]
    new_samples = [(x[0], 0 if x[1] == nrm_cls_idx else 1) for x in subset.samples]
    subset.class_name = [idx_to_class[x] for x in subset.targets]
    subset.targets = new_targets
    subset.samples = new_samples
    subset.imgs = new_samples
    return subset


class AddMask():
    def __init__(self, config):
        self.flist = list(glob.glob(config.TRAIN_MASK_FLIST + '/*.jpg')) + list(glob.glob(config.TRAIN_MASK_FLIST + '/*.png'))
        self.flist.sort()
        self.mask_set = []
        self.mask_type = config.MASK_TYPE
        for scale in config.SCALES:
            for mask_index in range(scale*4, (scale+1)*4):
                mask = Image.open(self.flist[mask_index])
                mask = transforms.Resize(config.INPUT_SIZE, interpolation=Image.NEAREST)(mask)
                # mask = (mask > 0).astype(np.uint8) * 255
                self.mask_set.append(transforms.ToTensor()(mask))

    def append_mask(self, batch):
        masks = []
        imgs = []
        label = []
        for i in range(len(batch)):
            img, target = batch[i]
            imgs.append(img)
            masks.append(random.choice(self.mask_set))
            label.append(target)
        imgs = torch.stack(imgs, dim=0)
        mask_batch = torch.stack(masks, dim=0)
        label = torch.FloatTensor(label)
        if self.mask_type == 0:
            mask_batch = None
        return imgs, mask_batch, label
