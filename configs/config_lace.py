
MODE = 1             # 1: train, 2: test, 3: eval
MASK_TYPE = 3
SEED = 10            # random seed
GPU = [0]            # list of gpu ids
DEBUG = 0            # turns on debugging mode
VERBOSE = 0          # turns on verbose mode in the output console

dataroot = './dataset/LaceAD'
workers = 4
normal_class = 'good'

TRAIN_MASK_FLIST = './mask'
TEST_MASK_FLIST = './mask'


LR = 0.0001                    # learning rate
D2G_LR = 0.1                   # discriminator/generator learning rate ratio
BETA1 = 0.0                    # adam optimizer beta1
BETA2 = 0.9                    # adam optimizer beta2
BATCH_SIZE = 4                 # input batch size for training
INPUT_SIZE = 512               # input image size for training 0 for original size
INPUT_CHANNELS = 3
SCALES = [1,2,3]
MAX_EPOCHS = 200                # maximum number of iterations to train the model

REC_LOSS_WEIGHT = 1             # l1 loss weight
FM_LOSS_WEIGHT = 0            # feature-matching loss weight
INPAINT_ADV_LOSS_WEIGHT = 0.001  # adversarial loss weight

GAN_LOSS = 'nsgan'               # nsgan | lsgan | hinge

LOG_INTERVAL = 10            # how many iterations to wait before logging training status (0: never)

STAGE = [1]
DATASET = 'LaceAD'
SUB_SET = ''
PATH = './ckpt/lace'
DEBUG = 0
