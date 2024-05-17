# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 13:06:56 2018

@author: norbot
"""

import torch
import torch.nn as nn


class Wighted_L1_Loss(torch.nn.Module):
    def __init__(self):
        super(Wighted_L1_Loss, self).__init__()

    def forward(self, pred, label):
        assert not torch.isnan(pred).any(), "Pred contains NaN" # 出现的是这个问题
        assert not torch.isinf(pred).any(), "Pred contains inf"
        assert not torch.isnan(label).any(), "Label contains NaN"
        assert not torch.isinf(label).any(), "Label contains inf"

        label_mask = label > 0.001
        _pred = pred[label_mask]
        _label = label[label_mask]
        n_valid_element = _label.size(0)
        if n_valid_element == 0:return 0
        diff_mat = torch.abs(_pred-_label)
        # diff_mat = torch.pow(_pred - _label, 2)
        loss = torch.sum(diff_mat)/(n_valid_element+1e-8)
        return loss

    def forward_depth(self, refined_sparse, sparse_depth, groundtruth):
        sparse_mask = sparse_depth > 0.1
        _refined_sparse = refined_sparse[sparse_mask]
        _groundtruth = groundtruth[sparse_mask]
        n_valid_element = _groundtruth.size(0)
        # if n_valid_element == 0:
        #     return 0
        diff_mat = torch.abs(_refined_sparse - _groundtruth)
        # diff_mat = torch.abs()
        loss = torch.sum(diff_mat) / n_valid_element
        return loss
    
    def forward_full(self, refined_sparse, sparse_depth, groundtruth):
        sparse_mask = sparse_depth > 0.1
        _refined_sparse = refined_sparse[sparse_mask]
        _groundtruth = groundtruth[sparse_mask]
        n_valid_element = _groundtruth.size(0)
        if n_valid_element == 0:
            return 0
        diff_mat = torch.abs(_refined_sparse - _groundtruth)
        # diff_mat = torch.abs()
        loss = torch.sum(diff_mat) / n_valid_element
        return loss
