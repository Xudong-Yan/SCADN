import os
import cv2
import random
import numpy as np
import torch
import argparse
from mmcv import Config
from src.experiments import ExpMvtec


def main(mode=None):
    r"""starts the model

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    config = load_config(mode)


    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)


    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    model = ExpMvtec(config)
    model.load()



    # model training
    if config.MODE == 1:
        print(config)
        print('\nstart training...\n')
        model.train()

    # model test
    elif config.MODE == 2:
        model.update_norm()
        print('\nstart testing...\n')
        model.test()

    # eval mode
    else:
        print('\nstart eval...\n')
        model.eval()


def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='the path of config file')
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints', help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--dataset', type=str, help='Dataset name:CIFAR|MvtecAD')
    parser.add_argument('--subset', type=str, help='subset name (default: None)')
    parser.add_argument('--gpu', type=str, help='gpu list')
    parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
    parser.add_argument('--val', type=str, help='path to the val images directory')
    parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')
    parser.add_argument('--output', type=str, help='path to the output directory')
    parser.add_argument('--debug', type=int, help='if not 0 will save debug image')
    parser.add_argument('--stage', type=int, help='if not 0 will save debug image')

    args = parser.parse_args()

    # load config file
    config = Config.fromfile(args.config_file)

    # any mode
    if args.dataset is not None:
        config.append('DATASET', args.dataset)
    if args.subset is not None:
        config.SUB_SET = args.subset
    if args.gpu is not None:
        config.GPU = list(args.gpu)
    if args.stage is not None:
        config.STAGE = [args.stage]

    if config.SUB_SET is not None:
        config.SUB_SET = str(config.SUB_SET)
        config.PATH = os.path.join(config.PATH, config.SUB_SET)

    # create checkpoints path if does't exist
    if not os.path.exists(config.PATH):
        os.makedirs(config.PATH)



    if mode == 1:
        config.MODE = 1
        if args.input is not None:
            config.append('TRAIN_FLIST', args.input)

        if args.mask is not None:
            config.append('TRAIN_MASK_FLIST', args.mask)

        if args.val is not None:
            config.append('VAL_FLIST', args.val)


    # test mode
    elif mode == 2:
        config.MODE = 2
        # config.INPUT_SIZE = 0

        if args.input is not None:
            config.TEST_FLIST = args.input

        if args.mask is not None:
            config.TEST_MASK_FLIST = args.mask

        if args.output is not None:
            config.RESULTS = args.output

        if args.debug is not None:
            config.DEBUG = args.debug


    # train mode

    return config


if __name__ == "__main__":
    main()
