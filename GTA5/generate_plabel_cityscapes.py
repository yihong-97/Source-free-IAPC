import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
import re
from packaging import version

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab_advent_no_p import get_deeplab_v2
from dataset.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
from PIL import Image
from utils.tool import fliplr
import matplotlib.pyplot as plt
import torch.nn as nn
import yaml

torch.backends.cudnn.benchmark = True

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

DATA_DIRECTORY = '../../../../../datasets/seg/Cityscapes'
DATA_LIST_PATH = 'dataset/cityscapes_list/train.txt'
SAVE_PATH = './data/Cityscapes/Pseudo_labels/train'

if not os.path.isdir(SAVE_PATH[:-6]):
    os.mkdir(SAVE_PATH[:-6])
    os.mkdir(SAVE_PATH)

IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 2975
BATCH_SIZE = 2
RESTORE_FROM = './pretrained/sourcemodel_gta5_res.pth'
SET = 'train'
MODEL = 'DeepLab'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30, 220, 220,
           0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70, 0, 60, 100, 0, 80,
           100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL, help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM, help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")
    parser.add_argument("--batchsize", type=int, default=BATCH_SIZE, help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET, help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH, help="Path to save result.")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()
    gpu0 = args.gpu
    batchsize = args.batchsize

    if args.model == 'DeepLab':
        model = get_deeplab_v2(num_classes=19, multi_level=False)

    if args.restore_from[:4] == 'http':
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from, map_location='cuda:0')['state_dict']

    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        i_parts = i.split('.')
        if not i_parts[1] == 'layer5':
            if i_parts[0] == 'module':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            else:
                new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
    model.load_state_dict(new_params)

    model.eval()
    model.cuda(gpu0)

    testloader = data.DataLoader(
        cityscapesDataSet(args.data_dir, args.data_list, crop_size=(512, 1024), resize_size=(1024, 512), mean=IMG_MEAN,
                          scale=False, mirror=False, set=args.set), batch_size=batchsize, shuffle=False,
        pin_memory=True, num_workers=4)

    scale = 1.25
    testloader2 = data.DataLoader(
        cityscapesDataSet(args.data_dir, args.data_list, crop_size=(round(512 * scale), round(1024 * scale)),
                          resize_size=(round(1024 * scale), round(512 * scale)), mean=IMG_MEAN, scale=False,
                          mirror=False, set=args.set), batch_size=batchsize, shuffle=False, pin_memory=True,
        num_workers=4)

    scale = 0.75
    testloader3 = data.DataLoader(
        cityscapesDataSet(args.data_dir, args.data_list, crop_size=(round(512 * scale), round(1024 * scale)),
                          resize_size=(round(1024 * scale), round(512 * scale)), mean=IMG_MEAN, scale=False,
                          mirror=False, set=args.set), batch_size=batchsize, shuffle=False, pin_memory=True,
        num_workers=4)

    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(1024, 2048), mode='bilinear')

    sm = torch.nn.Softmax(dim=1)

    for index, img_data in enumerate(zip(testloader, testloader2, testloader3)):
        batch, batch2, batch3 = img_data
        image, _, _, name = batch
        image2, _, _, name2 = batch2
        image3, _, _, name3 = batch3

        inputs = image.cuda()
        inputs2 = image2.cuda()
        inputs3 = image3.cuda()
        print('\r>>>>Extracting feature...%04d/%04d' % (index * batchsize, NUM_STEPS), end='')
        if args.model == 'DeepLab':
            with torch.no_grad():
                output1, output2 = model(inputs)
                output_batch = interp(sm(output2))
                output1, output2 = model(fliplr(inputs))
                output2 = fliplr(output2)
                output_batch += interp(sm(output2))

                output1, output2 = model(inputs2)
                output_batch += interp(sm(output2))
                output1, output2 = model(fliplr(inputs2))
                output2 = fliplr(output2)
                output_batch += interp(sm(output2))

                output1, output2 = model(inputs3)
                output_batch += interp(sm(output2))
                output1, output2 = model(fliplr(inputs3))
                output2 = fliplr(output2)
                output_batch += interp(sm(output2))

                del output1, output2, inputs2, inputs3
                output_batch = output_batch.cpu().data.numpy()

        output_batch = output_batch.transpose(0, 2, 3, 1)
        output_batch = np.asarray(np.argmax(output_batch, axis=3), dtype=np.uint8)
        for i in range(output_batch.shape[0]):
            output = output_batch[i, :, :]
            output_col = colorize_mask(output)
            output = Image.fromarray(output)
            name_tmp = name[i].split('/')[-1]
            dir_name = name[i].split('/')[-2]
            save_path = args.save + '/' + dir_name
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            output.save('%s/%s' % (save_path, name_tmp))
            output_col.save('%s/%s_color.png' % (save_path, name_tmp.split('.')[0]))

    return args.save


if __name__ == '__main__':
    with torch.no_grad():
        save_path = main()
    os.system('python compute_iou.py ../../../../../datasets/seg/Cityscapes/gtFine/train %s' % SAVE_PATH)
