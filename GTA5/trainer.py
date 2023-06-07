import torch.nn as nn
from torch.utils import data, model_zoo
import torch.optim as optim
import torch.nn.functional as F
from model.deeplab_advent import get_deeplab_v2
import torch
import torch.nn.init as init
import copy
import numpy as np
from utils.func import per_class_iu, fast_hist
from torchvision import transforms
from PIL import Image
import os
import math


palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


def train_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip

class AD_Trainer(nn.Module):
    def __init__(self, args):
        super(AD_Trainer, self).__init__()
        self.fp16 = args.fp16
        self.class_balance = args.class_balance
        self.often_balance = args.often_balance
        self.num_classes = args.num_classes

        ### lambda
        self.lambda_loss_pseudo_label = args.lambda_loss_pseudo_label
        self.lambda_loss_clustering_label = args.lambda_loss_clustering_label
        self.lambda_loss_clustering_sym_label = args.lambda_loss_clustering_sym_label
        self.lambda_loss_prediction_self = args.lambda_loss_prediction_self

        ###
        self.class_weight = torch.FloatTensor(self.num_classes).zero_().cuda() + 1
        self.often_weight = torch.FloatTensor(self.num_classes).zero_().cuda() + 1
        self.class_w = torch.FloatTensor(self.num_classes).zero_().cuda() + 1
        self.multi_gpu = args.multi_gpu
        self.snapshot_dir = args.snapshot_dir
        self.only_hard_label = args.only_hard_label
        self.interp = nn.Upsample(size=args.crop_size, mode='bilinear', align_corners=True)
        self.interp_target = nn.Upsample(size=args.crop_size, mode='bilinear', align_corners=True)
        self.interp_label = nn.Upsample(size=args.crop_size, mode='nearest')
        self.max_value = args.max_value

        ### init model
        if args.model == 'DeepLab':
            self.G = get_deeplab_v2(num_classes=self.num_classes, multi_level=False)
            if args.restore_from[:4] == 'http':
                saved_state_dict = model_zoo.load_url(args.restore_from)
            else:
                saved_state_dict = torch.load(args.restore_from, map_location='cuda:0')['state_dict']
                new_params = self.G.state_dict().copy()
                for i in saved_state_dict:
                    i_parts = i.split('.')
                    if not i_parts[1] == 'layer5':
                        if i_parts[0] == 'module':
                            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                        else:
                            new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
        self.G.load_state_dict(new_params)

        ### IAPC framework
        self.G_source = get_deeplab_v2(num_classes=self.num_classes, multi_level=False)
        self.G_source.load_state_dict(self.G.state_dict().copy())
        self.G_source_eval = get_deeplab_v2(num_classes=self.num_classes, multi_level=False)
        self.G_source_eval.load_state_dict(self.G.state_dict().copy())
        self.G_memory = get_deeplab_v2(num_classes=self.num_classes, multi_level=False)
        self.G_memory.load_state_dict(self.G.state_dict().copy())

        ### optimize
        self.gen_opt = optim.SGD(self.G.optim_parameters(args), lr=args.learning_rate, momentum=args.momentum,
                                 nesterov=True, weight_decay=args.weight_decay)
        self.G = self.G.cuda()
        self.G_source = self.G_source.cuda()
        self.G_source_eval = self.G_source_eval.cuda().eval()
        self.G_memory = self.G_memory.cuda()
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.seg_loss_fix = nn.CrossEntropyLoss(ignore_index=255)
        self.seg_val_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.sm = torch.nn.Softmax(dim=1)
        self.log_sm = torch.nn.LogSoftmax(dim=1)

    def update_class_criterion(self, labels):
        weight = torch.FloatTensor(self.num_classes).zero_().cuda()
        weight += 1
        count = torch.FloatTensor(self.num_classes).zero_().cuda()
        often = torch.FloatTensor(self.num_classes).zero_().cuda()
        often += 1
        n, h, w = labels.shape
        for i in range(self.num_classes):
            count[i] = torch.sum(labels == i)
            if count[i] < 64 * 64 * n:
                weight[i] = self.max_value
        if self.often_balance:
            often[count == 0] = self.max_value

        self.often_weight = 0.9 * self.often_weight + 0.1 * often
        self.class_weight = weight * self.often_weight
        return nn.CrossEntropyLoss(weight=self.class_weight, ignore_index=255, reduction='none')

    def update_source_memory_network(self):
        for param_q, param_k in zip(self.G.parameters(), self.G_memory.parameters()):
            param_k.data = param_k.data.clone() * 0.9999 + param_q.data.clone() * (1. - 0.9999)
        for buffer_q, buffer_k in zip(self.G.buffers(), self.G_memory.buffers()):
            buffer_k.data = buffer_q.data.clone()

    def update_centroids_label(self, feature_memory, label_memory, feature, label, label_contrast):

        label_memory = label_memory.clone()
        label_memory = F.interpolate(label_memory.type(torch.FloatTensor).unsqueeze(1), size=feature.size()[2:],
                                     mode='nearest')
        label = label.clone()
        label = F.interpolate(label.type(torch.FloatTensor).unsqueeze(1), size=feature.size()[2:], mode='nearest')
        label_contrast = label_contrast.clone()
        label_contrast = F.interpolate(label_contrast.type(torch.FloatTensor).unsqueeze(1), size=feature.size()[2:],
                                       mode='nearest')
        feature_memory = feature_memory.permute(0, 2, 3, 1).contiguous()
        feature = feature.permute(0, 2, 3, 1).contiguous()

        clu_prediction_list = []
        clu_probability_list = []
        for i in range(feature_memory.size(0)):
            label_memory_i = label_memory[i]
            feature_memory_i = feature_memory[i]
            feature_i = feature[i]

            label_memory_i = label_memory_i.view(-1)
            feature_memory_i = feature_memory_i.view(-1, feature_memory_i.size(-1))
            feature_i = feature_i.view(-1, feature_i.size(-1))

            centroid_i_list = []
            indices = [torch.tensor(i_c).cuda() for i_c in range(self.num_classes)]
            for j in range(self.num_classes):
                mask_i_j = torch.eq(label_memory_i, j)  # size Ns
                feature_i_j = feature_memory_i[mask_i_j, :]  # size Ns_i x F
                if feature_i_j.size(0) > 0:
                    centroid_i_j = torch.mean(feature_i_j, dim=0, keepdim=True)  # size 1 x F
                    centroid_i_list.append(centroid_i_j)
                else:
                    centroid_i_list.append(torch.tensor([[0.] * feature_memory_i.size(1)], dtype=torch.float).cuda())
                    indices.remove(j)
            centroids_i = torch.stack(centroid_i_list, dim=0).squeeze(1)

            feat_i = F.normalize(feature_i, p=2, dim=1)
            centroids_i = F.normalize(centroids_i.detach(), p=2, dim=1)

            logits = feat_i.mm(centroids_i.permute(1, 0).contiguous())
            logits_pro = logits
            clu_probability_list.append(logits_pro)
            logits_arg = torch.argmax(logits, dim=1)
            clu_prediction_list.append(logits_arg)

        clu_predictions = torch.stack(clu_prediction_list, dim=0)
        clu_probabilitys = torch.stack(clu_probability_list, dim=0)
        clu_predictions = clu_predictions.view(label_contrast.size())
        clu_probabilitys = clu_probabilitys.view(label_contrast.size(0), self.num_classes, label_contrast.size(2),
                                                 label_contrast.size(3))

        return clu_predictions.squeeze(1), clu_probabilitys

    def gen_update(self, images, images_t, labels, labels_t, i_iter, image_aug1, label_aug1, image_aug2, label_aug2):
        self.update_source_memory_network()

        loss_dicts = {}
        self.gen_opt.zero_grad()
        loss_total = 0.0
        with torch.no_grad():

            pred1_source, pred2_source, _ = self.G_source_eval(image_aug1)
            pred_source_up = self.interp_target(pred2_source)
            pred1_memory, pred2_memory, feature_memory = self.G_memory(images)
            pred2_memory_up = self.interp_target(pred2_memory)
            pred_memory_label_up = torch.argmax((pred2_memory_up), dim=1)

        pred1_aug, pred2_aug, feature_aug = self.G(image_aug2)
        pred2_aug_up = self.interp_target(pred2_aug)
        pred_aug_up = pred2_aug_up

        pred_aug_label = torch.argmax((pred2_aug), dim=1)
        pred_aug_label_up = torch.argmax((pred2_aug_up), dim=1)
        predictions_dicts = {'pred_source_up': pred_source_up, 'pred_aug_up': pred_aug_up}

        if self.class_balance:
            self.seg_loss = self.update_class_criterion(label_aug2)

        if self.lambda_loss_pseudo_label is not None:
            source_pro_two = torch.argsort(pred_source_up, dim=1)[:, -2, :, :]
            source_pro_one = torch.argmax(pred_source_up, dim=1)
            source_pro_one = torch.gather(pred_source_up, 1, source_pro_one.unsqueeze(1))
            source_pro_two = torch.gather(pred_source_up, 1, source_pro_two.unsqueeze(1))
            entropy_source = (1. - torch.div(source_pro_two, source_pro_one)).squeeze(1)
            loss_seg2 = self.seg_loss(pred2_aug_up, label_aug2)
            loss_pse = torch.mean(loss_seg2 * entropy_source)
            loss_total += self.lambda_loss_pseudo_label * loss_pse
            loss_dicts['loss_pseudo_label'] = self.lambda_loss_pseudo_label * loss_pse

        if self.lambda_loss_clustering_label is not None or self.lambda_loss_clustering_sym_label is not None:
            clu_label, clu_pro = self.update_centroids_label(feature_memory, pred_memory_label_up, feature_aug,
                                                             pred_aug_label_up, label_aug2)
            clu_label_up = F.interpolate(clu_label.detach().type(torch.FloatTensor).cuda().unsqueeze(1),
                                         size=label_aug2.size()[1:], mode='nearest').squeeze(1)

        if self.lambda_loss_clustering_sym_label is not None:
            loss_seg2_clu = self.seg_loss_fix(pred2_aug_up, clu_label_up.long())
            loss_clu = torch.mean(loss_seg2_clu)
            clu_pro_up = self.interp_target(clu_pro)
            loss_clu_pro = self.seg_loss_fix(clu_pro_up, pred_aug_label_up)
            loss_clu_pro = torch.mean(loss_clu_pro)
            loss_sym_label = loss_clu_pro + loss_clu
            loss_total += self.lambda_loss_clustering_sym_label * loss_sym_label
            loss_dicts['loss_clustering_sym_label_label'] = self.lambda_loss_clustering_sym_label * loss_clu
            loss_dicts['loss_clustering_sym_label_probability'] = self.lambda_loss_clustering_sym_label * loss_clu_pro

        if self.lambda_loss_clustering_label is not None:
            clu_label_mask = torch.ne(clu_label_up, label_aug2)
            clu_label_up_mask = clu_label_up.clone()
            clu_label_up_mask[clu_label_mask] = 255
            loss_seg2_clu_mask = self.seg_loss(pred2_aug_up, clu_label_up_mask.long())
            loss_clu_mask = torch.mean(loss_seg2_clu_mask)
            loss_total += self.lambda_loss_clustering_label * loss_clu_mask
            loss_dicts['loss_clustering_label'] = self.lambda_loss_clustering_label * loss_clu_mask

        if self.lambda_loss_prediction_self is not None:
            epsilon = 1e-5
            loss_self2 = torch.sum(-self.sm(pred2_aug_up) * torch.log(self.sm(pred2_aug_up) + epsilon), dim=1)
            loss_self = torch.mean(loss_self2)
            loss_total += self.lambda_loss_prediction_self * loss_self
            loss_dicts['loss_prediction_self'] = self.lambda_loss_prediction_self * loss_self

        loss_total.backward()
        self.gen_opt.step()

        pred1_target, pred2_target, _ = self.G(images_t)
        pred2_target = self.interp_target(pred2_target)
        val_loss = torch.mean(self.seg_val_loss(pred2_target, labels_t))


        return predictions_dicts, loss_dicts, loss_total, val_loss

    def evaluate_target(self, test_loader, cfg, fixed_test_size=True, per_class = False,verbose=False):
        device = cfg.GPU_ID
        interp = None
        if fixed_test_size:
            interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]), mode='bilinear',
                                 align_corners=True)
        self.G.eval()
        hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
        print('evaluation nums: ', len(test_loader))
        for index, batch in (enumerate(test_loader)):
            image, label, _, name = batch
            if not fixed_test_size:
                interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
            with torch.no_grad():
                output = None
                pred_aux, pred_main, _ = self.G(image.cuda(device))
                output = interp(pred_main).cpu().data[0].numpy()

                assert output is not None, 'Output is None'
                output = output.transpose(1, 2, 0)
                output = np.argmax(output, axis=2)
            label = label.numpy()[0]
            hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
        inters_over_union_classes = per_class_iu(hist)
        if per_class:
            name_classes = ["road", "sidewalk", "building", "wall", "fence", "pole", "light", "sign", "vegetation",
                            "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motocycle", "bicycle"]
            for ind_class in range(self.num_classes):
                print(('===>' + name_classes[ind_class] + ':\t' + str(
                    round(inters_over_union_classes[ind_class] * 100, 2))))
        print(f'mIoU = \t{round(np.nanmean(inters_over_union_classes) * 100, 2)}')
        self.G.train()
        return round(np.nanmean(inters_over_union_classes) * 100, 2)

