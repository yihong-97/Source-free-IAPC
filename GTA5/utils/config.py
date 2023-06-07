import os.path as osp

import numpy as np
from easydict import EasyDict
import pathlib

project_root = pathlib.Path(__file__).resolve().parents[1]


cfg = EasyDict()

# COMMON CONFIGS
# source domain
cfg.SOURCE = 'GTA'
# target domain
cfg.TARGET = 'Cityscapes'
# Number of workers for dataloading
cfg.NUM_WORKERS = 4
# List of training images
cfg.DATA_LIST_SOURCE = str(project_root / 'dataset/gta5_list/{}.txt')
cfg.DATA_LIST_TARGET = str(project_root / 'dataset/cityscapes_list/{}.txt')
# Directories
DATA_DIR = '../../../../../datasets/seg'
cfg.DATA_DIRECTORY_SOURCE = DATA_DIR + '/GTA5'
cfg.DATA_DIRECTORY_TARGET = DATA_DIR + '/Cityscapes'
# Number of object classes
cfg.NUM_CLASSES = 19
# Exp dirs
cfg.EXP_NAME = ''
cfg.EXP_ROOT = project_root / 'experiments'
cfg.EXP_ROOT_SNAPSHOT = osp.join(cfg.EXP_ROOT, 'snapshots')
cfg.EXP_ROOT_LOGS = osp.join(cfg.EXP_ROOT, 'logs')
# CUDA
cfg.GPU_ID = 0

# TRAIN CONFIGS
cfg.TRAIN = EasyDict()
cfg.TRAIN.SET_SOURCE = 'all'
cfg.TRAIN.drop_model = 0
cfg.TRAIN.fixed_conv1 = 0
cfg.TRAIN.SET_TARGET = 'train'
cfg.TRAIN.BATCH_SIZE_SOURCE = 1
cfg.TRAIN.BATCH_SIZE_TARGET = 1
cfg.TRAIN.IGNORE_LABEL = 255
cfg.TRAIN.INPUT_SIZE_SOURCE = (1280, 720)
cfg.TRAIN.INPUT_SIZE_TARGET = (1024, 512)
# cfg.TRAIN.INPUT_SIZE_TARGET = (1280, 640)
cfg.TRAIN.INPUT_SIZE_TARGET_bdds = (1280, 720)
cfg.TRAIN.INPUT_SIZE_TARGET_vistass = (1024, 768)
# Class info
cfg.TRAIN.INFO_SOURCE = ''
cfg.TRAIN.INFO_TARGET = str(project_root / 'dataset/cityscapes_list/info.json')
# Segmentation network params
cfg.TRAIN.MODEL = 'DeepLabv2'
cfg.TRAIN.MULTI_LEVEL = True
cfg.TRAIN.RESTORE_FROM = '../pretrained_models/DeepLab_resnet_pretrained_imagenet.pth'
cfg.TRAIN.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TRAIN.LEARNING_RATE = 2.5e-4
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 0.0005
cfg.TRAIN.POWER = 0.9
cfg.TRAIN.LAMBDA_SEG_MAIN = 1.0
cfg.TRAIN.LAMBDA_SEG_AUX = 0.1  # weight of conv4 prediction. Used in multi-level setting.
# Domain adaptation
cfg.TRAIN.DA_METHOD = 'Source_eval_t'  #{Source_eval_t, Source}
# Adversarial training params
cfg.TRAIN.LEARNING_RATE_D = 1e-4
cfg.TRAIN.LAMBDA_ADV_MAIN = 0.001
cfg.TRAIN.LAMBDA_ADV_AUX = 0.0002
# MinEnt params
cfg.TRAIN.LAMBDA_ENT_MAIN = 0.001
cfg.TRAIN.LAMBDA_ENT_AUX = 0.0002
# Other params
cfg.TRAIN.MAX_ITERS = 250000
cfg.TRAIN.EARLY_STOP = 250000
cfg.TRAIN.SAVE_PRED_EVERY = 2000
cfg.TRAIN.SHOW_LOSS_EVERY = 100
cfg.TRAIN.PRED_T_EVERY = 2000
cfg.TRAIN.SNAPSHOT_DIR = ''
cfg.TRAIN.RANDOM_SEED = 1234
cfg.TRAIN.TENSORBOARD_LOGDIR = ''
cfg.TRAIN.TENSORBOARD_VIZRATE = 100

# TEST CONFIGS
cfg.TEST = EasyDict()
cfg.TEST.MODE = 'single'  # {'single', 'best'}
# model
cfg.TEST.MODEL = ('DeepLabv2',)
cfg.TEST.MODEL_WEIGHT = (1.0,)
cfg.TEST.MULTI_LEVEL = (True,)
cfg.TEST.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TEST.RESTORE_FROM = ('../../../HCL/pretrained_models/GTA5_HCL_source.pth',)
cfg.TEST.SNAPSHOT_DIR = ('',)  # used in 'best' mode
cfg.TEST.SNAPSHOT_STEP = 2000  # used in 'best' mode
cfg.TEST.SNAPSHOT_MAXITER = 250000  # used in 'best' mode
# Test sets
cfg.TEST.SET_TARGET = 'val'
cfg.TEST.SET_SOURCE = 'val'
cfg.TEST.TEST_ON_GTA = False
cfg.TEST.BATCH_SIZE_SOURCE = 1
cfg.TEST.BATCH_SIZE_TARGET = 1
cfg.TEST.INPUT_SIZE_SOURCE = (1280, 720)
cfg.TEST.INPUT_SIZE_TARGET = (1024, 512)
cfg.TEST.INPUT_SIZE_TARGET_bdds = (1280, 720)
cfg.TEST.INPUT_SIZE_TARGET_vistass = (1024, 768)
cfg.TEST.OUTPUT_SIZE_TARGET = (2048, 1024)
cfg.TEST.OUTPUT_SIZE_TARGET_bdds = (1280, 720)
cfg.TEST.OUTPUT_SIZE_TARGET_vistass = (2048, 1024)
cfg.TEST.INFO_TARGET = str(project_root / 'dataset/cityscapes_list/info.json')
cfg.TEST.INFO_TARGET_Bdds = str(project_root / 'dataset/bdds_list/info.json')
cfg.TEST.INFO_TARGET_vistass = str(project_root / 'dataset/vistass_list/info.json')
cfg.TEST.WAIT_MODEL = True

# frequency analysis
cfg.TRAIN.F_select = False
cfg.TRAIN.F_component = [0, 1]

# cfg.TEST.F_select = False
# cfg.TEST.F_component = [0.0, 1.0]

cfg.TEST.class_names_gta = ["road",
                            "sidewalk",
                            "building",
                            "wall",
                            "fence",
                            "pole",
                            "light",
                            "sign",
                            "vegetation",
                            "terrain",
                            "sky",
                            "person",
                            "rider",
                            "car",
                            "truck",
                            "bus",
                            "train",
                            "motocycle",
                            "bicycle"]


