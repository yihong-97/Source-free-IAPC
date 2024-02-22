"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple

class GraphConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07*2):
        super(GraphConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        dim_in = 2048
        feat_dim = 2048
        self.head_1 = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        self.head_2 = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

    def forward(self, t_feat, s_feat, graph_cn, labels=None, mask=None):    

        qx = graph_cn.graph.wq(s_feat)
        kx = graph_cn.graph.wk(s_feat)        
        sim_mat = qx.matmul(kx.transpose(-1, -2))
        dot_mat = sim_mat.detach().clone()

        thresh = 0.5
        dot_mat -= dot_mat.min(1, keepdim=True)[0]
        dot_mat /= dot_mat.max(1, keepdim=True)[0]
        mask = ((dot_mat>thresh)*1).detach().clone()
        mask.fill_diagonal_(1)

        anchor_feat = self.head_1(s_feat)
        contrast_feat = self.head_2(s_feat)

        anchor_feat = F.normalize(anchor_feat, dim=1)
        contrast_feat = F.normalize(contrast_feat, dim=1)

        ss_anchor_dot_contrast = torch.div(torch.matmul(anchor_feat, contrast_feat.T), self.temperature)  ##### torch.Size([6, 6])
        logits_max, _ = torch.max(ss_anchor_dot_contrast, dim=1, keepdim=True)  ##### torch.Size([6, 1]) - contains max value along dim=1
        ss_graph_logits = ss_anchor_dot_contrast - logits_max.detach()

        ss_graph_all_logits = torch.exp(ss_graph_logits)
        ss_log_prob = ss_graph_logits - torch.log(ss_graph_all_logits.sum(1, keepdim=True))
        ss_mean_log_prob_pos = (mask * ss_log_prob).sum(1) / mask.sum(1)  
    
        # loss
        ss_loss = - (self.temperature / self.base_temperature) * ss_mean_log_prob_pos
        ss_loss = ss_loss.mean()

        return ss_loss


class Loss_PC(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07 * 2):
        super(GraphConLoss_PC, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        dim_in = 2048
        feat_dim = 2048
        self.head_1 = nn.Sequential(nn.Linear(dim_in, dim_in), nn.ReLU(inplace=True), nn.Linear(dim_in, feat_dim))
        self.head_2 = nn.Sequential(nn.Linear(dim_in, dim_in), nn.ReLU(inplace=True), nn.Linear(dim_in, feat_dim))

    def forward(self, t_feat, t_logi, s_feat, s_logi, graph_cn, labels=None, mask=None):
        s_feat = self.head_1(s_feat)
        t_feat = self.head_2(t_feat)
        ## prototypes
        lambda_ps = 0.4
        lambda_pe = 1.
        t_logi_arg = torch.argmax(t_logi[0], dim=1)
        t_centroid_list = []
        for i in range(9):
            mask_i = torch.eq(t_logi_arg, i)
            t_feat_i = t_feat[mask_i]

            if t_feat_i.size(0) > 0:
                t_centroid_i = torch.mean(t_feat_i, dim=0, keepdim=True)  # size 1 x F
                t_centroid_list.append(t_centroid_i)
            else:
                t_centroid_list.append(torch.tensor([[0.] * t_feat.size(1)], dtype=torch.float).cuda())
        t_centroids = torch.stack(t_centroid_list, dim=0).squeeze(1)
        s_feat_nor = F.normalize(s_feat, p=2, dim=1)
        t_centroids = F.normalize(t_centroids.detach(), p=2, dim=1)
        logits_centr = s_feat_nor.mm(t_centroids.permute(1, 0).contiguous())
        logits_centr_arg = torch.argmax(logits_centr, dim=1)

        ## Sym_CE
        s_logi_arg = torch.argmax(s_logi[0], dim=1)
        loss_sym_cs = cross_entropy(s_logi[0], logits_centr_arg, reduction='mean')
        loss_sym_sc = cross_entropy(logits_centr, s_logi_arg, reduction='mean')
        loss_sym = loss_sym_cs + loss_sym_sc

        label_sc_mask = torch.ne(s_logi_arg, logits_centr_arg)
        s_logi_arg_mask = s_logi_arg.clone()
        s_logi_arg_mask[label_sc_mask] = -1
        loss_pe = F.cross_entropy(s_logi[0], s_logi_arg_mask, ignore_index=-1, reduction='mean')

        return loss_sym*lambda_ps, loss_pe*lambda_pe


