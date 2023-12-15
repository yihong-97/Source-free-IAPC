import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
import time
import yaml
from tensorboardX import SummaryWriter
from utils.config import cfg
from trainer import AD_Trainer
from utils.loss import CrossEntropy2d
from utils.tool import adjust_learning_rate
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.cityscapes_test import CityscapesDataSet

from dataset.cityscapes_pseudo_dataset import cityscapes_pseudo_DataSet
# ##
# # Table IV - SP
# from dataset.cityscapes_pseudo_SP_dataset import cityscapes_pseudo_SP_DataSet
# ##



IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

AUTOAUG = True
AUTOAUG_TARGET = False

MODEL = 'DeepLab'
BATCH_SIZE = 6
ITER_SIZE = 1
NUM_WORKERS = 2
DATA_DIRECTORY = '../../../../../datasets/seg/Cityscapes'
DATA_LABEL_DIRECTORY = './data/Cityscapes/Pseudo_labels'
DATA_LIST_PATH = 'dataset/cityscapes_list/train.txt'
DROPRATE = 0.2
IGNORE_LABEL = 255
INPUT_SIZE = '1280,640'
DATA_DIRECTORY_TARGET = '../../../../../datasets/seg/Cityscapes'
DATA_LIST_PATH_TARGET = 'dataset/cityscapes_list/train.txt'
INPUT_SIZE_TARGET = '1024,512'
CROP_SIZE = '512, 256'
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
MAX_VALUE = 7
NUM_CLASSES = 19
NUM_STEPS = 100000
NUM_STEPS_STOP = 9000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './pretrained/sourcemodel_gta5_res.pth'
SAVE_NUM_IMAGES = 2
EVALUATE_EVERY = 1000
SAVE_EVALUATE = 0
SAVE_TB_EVERY = 100
THRESHOLD = 1.0
WEIGHT_DECAY = 0.0005
WARM_UP = 5000
LAMBDA_LOSS_IA = 0.2
LAMBDA_LOSS_PE = 0.5
LAMBDA_LOSS_PS = 0.04
LAMBDA_LOSS_IM = 2.

TARGET = 'cityscapes'
SET_TARGET = 'train'
TRAIN_METHOD = 'test'
ONLY_HARD_LABEL = 0



def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--autoaug", type=bool, default=AUTOAUG, help="use augmentation or not" )
    parser.add_argument("--autoaug_target", type=bool, default=AUTOAUG_TARGET, help="use augmentation or not" )
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-label-dir", type=str, default=DATA_LABEL_DIRECTORY,
                        help="Path to the directory containing the source dataset label.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--droprate", type=float, default=DROPRATE,
                        help="DropRate.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--crop-size", type=str, default=CROP_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")  # 没用
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--max-value", type=float, default=MAX_VALUE,
                        help="Max Value of Class Weight.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror_target", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--evaluate_every", type=int, default=EVALUATE_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--save_evaluate", type=int, default=SAVE_EVALUATE,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--save-tb-every", type=int, default=SAVE_TB_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=None,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--train-method", type=str, default=TRAIN_METHOD, help = 'warm up iteration')
    parser.add_argument("--warm-up", type=float, default=WARM_UP, help = 'warm up iteration')
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help = 'warm up iteration')
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--class-balance", action='store_true', default=True, help="class balance.")
    parser.add_argument("--use-se", action='store_true', default=True, help="use se block.")
    parser.add_argument("--only_hard_label",type=float, default=ONLY_HARD_LABEL,
                         help="class balance.")
    parser.add_argument("--train_bn", action='store_true', default=True, help="train batch normalization.")
    parser.add_argument("--sync_bn", action='store_true', help="sync batch normalization.")
    parser.add_argument("--often-balance", action='store_true', default=True, help="balance the apperance times.")
    parser.add_argument("--gpu-ids", type=str, default='0', help = 'choose gpus')
    parser.add_argument("--tensorboard", action='store_false', help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Path to the directory of log.")
    parser.add_argument("--set_target", type=str, default=SET_TARGET,
                        help="choose adaptation set.")
    parser.add_argument('-lpl',"--lambda_loss_pseudo_label", type=float, default=LAMBDA_LOSS_IA)
    parser.add_argument('-lcl',"--lambda_loss_clustering_label", type=float, default=LAMBDA_LOSS_PE)
    parser.add_argument('-lcsl',"--lambda_loss_clustering_sym_label", type=float, default=LAMBDA_LOSS_PS)
    parser.add_argument('-lps',"--lambda_loss_prediction_self", type=float, default=LAMBDA_LOSS_IM)
    return parser.parse_args()


args = get_arguments()
for arg in vars(args):
    print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))



def main():
    # INIT
    _init_fn = None
    if args.random_seed:
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

        def _init_fn(worker_id):
            np.random.seed(args.random_seed + worker_id)

    w, h = map(int, args.input_size.split(','))
    args.input_size = (w, h)

    w, h = map(int, args.crop_size.split(','))
    args.crop_size = (h, w)

    w, h = map(int, args.input_size_target.split(','))
    args.input_size_target = (w, h)

    cudnn.enabled = True
    cudnn.benchmark = True

    if args.snapshot_dir:
        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)
    else:
        EXP_NAME = time.strftime('%Y%m%d%H%M')+ f': {args.train_method}'
        args.snapshot_dir = os.path.join('./log', EXP_NAME)
        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

    with open('%s/opts.yaml' % args.snapshot_dir, 'w') as fp:
        yaml.dump(vars(args), fp, default_flow_style=False)

    if args.tensorboard:
        args.log_dir = os.path.join(args.snapshot_dir, 'tensorboard')
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        writer = SummaryWriter(args.log_dir)

    str_ids = args.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    num_gpu = len(gpu_ids)
    args.multi_gpu = False
    if num_gpu > 1:
        args.multi_gpu = True
        Trainer = AD_Trainer(args)
        Trainer.G = torch.nn.DataParallel(Trainer.G, gpu_ids)
    else:
        Trainer = AD_Trainer(args)

    trainloader = data.DataLoader(cityscapes_pseudo_DataSet(args.data_dir, args.data_label_dir, args.data_list,
                                                            max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                            resize_size=args.input_size, crop_size=args.crop_size,
                                                            scale=True, mirror=True, mean=IMG_MEAN, set='train',
                                                            autoaug=args.autoaug, threshold=args.threshold),
                                  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True, worker_init_fn=_init_fn)
    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     resize_size=args.input_size_target, crop_size=args.crop_size,
                                                     scale=False, mirror=args.random_mirror_target, mean=IMG_MEAN,
                                                     set=args.set_target, autoaug=args.autoaug_target),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True, drop_last=True)
    targetloader_iter = enumerate(targetloader)

    test_dataset = CityscapesDataSet(root=cfg.DATA_DIRECTORY_TARGET,
                                     list_path=cfg.DATA_LIST_TARGET,
                                     set=cfg.TEST.SET_TARGET,
                                     info_path=cfg.TEST.INFO_TARGET,
                                     crop_size=cfg.TEST.INPUT_SIZE_TARGET,
                                     mean=cfg.TEST.IMG_MEAN,
                                     labels_size=cfg.TEST.OUTPUT_SIZE_TARGET)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=cfg.TEST.BATCH_SIZE_TARGET,
                                  num_workers=cfg.NUM_WORKERS,
                                  shuffle=False,
                                  pin_memory=True)

    # ##
    # # Table IV - SP
    # SP_loader = data.DataLoader(cityscapes_pseudo_SP_DataSet(args.data_dir, args.data_label_dir, args.data_list,
    #                                                         max_iters=args.num_steps * args.iter_size * args.batch_size,
    #                                                         resize_size=args.input_size, crop_size=args.crop_size,
    #                                                         scale=True, mirror=True, mean=IMG_MEAN, set='train',
    #                                                         autoaug=args.autoaug, threshold=args.threshold),
    #                               batch_size=2, shuffle=False, num_workers=args.num_workers,
    #                               pin_memory=True, drop_last=True, worker_init_fn=_init_fn)
    # sp_have = Trainer.estimate_centroids(SP_loader)
    # ##

    for i_iter in range(args.num_steps):

        adjust_learning_rate(Trainer.gen_opt,  i_iter, args)

        print('\r>>>Current Iter step: %08d, Learning rate: %f'% (i_iter, Trainer.gen_opt.param_groups[0]['lr']), end='')

        # ##
        # # Table IV - for SP - Not using
        # if i_iter % 2970 == 0 and i_iter != 0:
        #     print('update prototypes')
        #     sp_have = Trainer.estimate_centroids(SP_loader)
        # ##
        for sub_i in range(args.iter_size):
            _, batch = trainloader_iter.__next__()
            _, batch_t = targetloader_iter.__next__()

            images, labels, _, name, image_aug1, label_label_aug1, image_aug2, label_label_aug2 = batch
            images = images.cuda()
            image_aug1 = image_aug1.cuda()
            image_aug2 = image_aug2.cuda()
            labels = labels.long().cuda()
            label_label_aug1 = label_label_aug1.long().cuda()
            label_label_aug2 = label_label_aug2.long().cuda()

            images_t, labels_t, _, name_t = batch_t
            images_t = images_t.cuda()
            labels_t = labels_t.long().cuda()

            predictions_dicts, loss_dicts, loss_total, val_loss = Trainer.gen_update(images, images_t, labels, labels_t, i_iter, image_aug1, label_label_aug1, image_aug2, label_label_aug2)


        del predictions_dicts

        if args.tensorboard:
            scalar_info = {
                'total_loss': loss_total,
                'val_loss': val_loss,
            }
            scalar_info.update(loss_dicts)

            if i_iter % args.save_tb_every == 0:
                for key, val in scalar_info.items():
                    writer.add_scalar(key, val, i_iter)

        del loss_total, val_loss
        del loss_dicts

        if i_iter >= args.num_steps_stop - 1:
            mIoU = Trainer.evaluate_target(test_loader, cfg, per_class=True)
            print('save model ...')
            torch.save(Trainer.G.state_dict(), osp.join(args.snapshot_dir, 'gta5_to_cityscapes_final.pth'))
            break



    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()
