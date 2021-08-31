import logging
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F


def setup_logger(output_dir, basename='log', distributed_rank=0):
    os.makedirs(output_dir, exist_ok=True)

    # ---- set up logger ----
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s: %(message)s", '%m%d%H%M%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if basename[-4:] == '.txt':
        basename = basename[:-4]
    txt_name = basename + '.txt'
    for i in range(2, int(1e10)):
        if os.path.exists(os.path.join(output_dir, txt_name)):
            txt_name = f'{basename}-{i}.txt'
        else:
            break
    fh = logging.FileHandler(os.path.join(output_dir, txt_name), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def log_info(log_str):
    logger = logging.getLogger()
    if len(logger.handlers):
        logger.info(log_str)
    else:
        print(log_str)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_num_units_bins_and_nums_for_plt(layers, model_name):
    num_bins_list, plt_row_list, plt_col_list, figsize_w_list, figsize_h_list = [], [], [], [], []
    num_units_list = []
    for layer in layers:
        if 'MFF' in model_name:
            if layer == 'D' or layer == 'MFF':
                num_units = 64
            elif layer == 'Rconv0' or layer == 'Rconv1':
                num_units = 128
            else:
                raise NotImplementedError
        elif 'BTS' in model_name:
            if layer == 'upconv3':
                num_units = 128
            elif layer == 'iconv2' or layer == 'upconv2':
                num_units = 64
            elif layer == 'iconv1' or layer == 'upconv1':
                num_units = 32
            else:
                raise NotImplementedError
        elif 'CSPN' in model_name:
            if layer == 'layer4':
                num_units = 64
            elif layer == 'layer3':
                num_units = 256
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        num_units_list.append(num_units)
    for num_units in num_units_list:
        DEFAULT_NUM_BINS = 64   # 暂时都用64个bin
        if num_units < DEFAULT_NUM_BINS:
            num_bins_list.append(num_units)
        else:
            num_bins_list.append(DEFAULT_NUM_BINS)
        if num_units == 64:
            plt_row, plt_col, figsize_h, figsize_w = 8, 8, 30, 50
        elif num_units == 32:
            plt_row, plt_col, figsize_h, figsize_w = 4, 8, 15, 50
        elif num_units == 128:
            plt_row, plt_col, figsize_h, figsize_w = 16, 8, 60, 50
        elif num_units == 256:
            plt_row, plt_col, figsize_h, figsize_w = 16, 16, 60, 100
        else:
            raise NotImplementedError
        plt_row_list.append(plt_row)
        plt_col_list.append(plt_col)
        figsize_h_list.append(figsize_h)
        figsize_w_list.append(figsize_w)

    return num_units_list, num_bins_list, plt_row_list, plt_col_list, figsize_w_list, figsize_h_list


def get_sid_thresholds(K, dataset_name):
    """
    :param K:
    :param dataset_name:
    :return: thresholds: a list whose length is K+1, thresholds[0]=0, thresholds[1]=alpha (0.75 for nyuv2), thresholds[K]=beta
    """
    if dataset_name == 'nyuv2':
        alpha, beta = 0.75, 10.
    elif dataset_name == 'kitti':
        alpha, beta = 3., 80.
    else:
        raise NotImplementedError
    thresholds = [0., alpha]
    for i in range(1, K-1):
        # ti = np.exp(np.log(alpha + (1 - alpha)) + (np.log((beta + (1 - alpha)) / (alpha + (1 - alpha))) * i) / (K-1)) - (1 - alpha)
        ti = np.exp((np.log(beta + (1 - alpha)) * i) / (K-1)) - (1 - alpha) # same but simpler
        thresholds.append(ti)
    thresholds.append(beta)
    return thresholds


def discretize_depths(depths, dataset_name, K=None, thresholds=None):
    discrete_depths = torch.zeros(depths.shape).to(depths.device)
    assert K or thresholds, "At least one of them should be given."
    if not thresholds:
        thresholds = get_sid_thresholds(K, dataset_name)
    else:
        if not K:
            K = len(thresholds) - 1
        else:
            assert K == len(thresholds) - 1
    discrete_depths[torch.where(depths <= 0)] = -1  # invalid depth -> -1
    for i in range(K):
        discrete_depths[torch.where((depths > thresholds[i]) & (depths <= thresholds[i+1]))] = i
    return discrete_depths


def count_discrete_depth(discrete_depth, discrete_depth_cnt=None):
    """

    :param discrete_depth: 4-dims: (img_i, 0, depth_h, depth_w)
    :param discrete_depth_cnt: 1-dim: bin_i
    :return: discrete_depth_cnt
    """
    device = torch.device("cpu") if isinstance(discrete_depth, np.ndarray) else discrete_depth.device
    if discrete_depth_cnt is None:
        discrete_depth_cnt = torch.zeros(1, dtype=torch.double).to(device)
    for img_i in range(len(discrete_depth)):
        bin_idx_max = int(discrete_depth[img_i].max())
        if bin_idx_max >= len(discrete_depth_cnt):  # need to extend discrete_depth_cnt
            discrete_depth_cnt = torch.cat([discrete_depth_cnt, torch.zeros(bin_idx_max - len(discrete_depth_cnt) + 1, dtype=torch.double).to(device)], axis=-1)
        for bin_idx_iter in range(bin_idx_max+1):
            cur_xy_tuples = torch.where(discrete_depth[img_i][0] == bin_idx_iter)
            if len(cur_xy_tuples[0]) == 0:
                continue
            discrete_depth_cnt[bin_idx_iter] += len(cur_xy_tuples[0])

    return discrete_depth_cnt


def add_fmaps_on_discrete_depths(discrete_depths, fmaps, discrete_depth_sum=None):
    """

    :param discrete_depths: 4-dims: (img_i, 0, depth_h, depth_w)
    :param fmaps: 4-dims: (img_i, unit_i, fmap_h, fmap_w)
    :param discrete_depth_sum: 2-dims: (unit_i, bin_i)
    :return: discrete_depth_sum
    """
    device = fmaps.device
    if discrete_depth_sum is None:
        discrete_depth_sum = torch.zeros((fmaps.shape[1], 1), dtype=torch.double).to(device)
    h, w = discrete_depths.shape[2], discrete_depths.shape[3]
    fmaps = F.interpolate(fmaps, (h, w), mode='bilinear', align_corners=False)
    for img_i in range(len(discrete_depths)):
        bin_idx_max = int(discrete_depths[img_i].max())
        if bin_idx_max >= discrete_depth_sum.shape[1]:     # need to extend discrete_depth_sum
            discrete_depth_sum = torch.cat([discrete_depth_sum, torch.zeros((discrete_depth_sum.shape[0], bin_idx_max - discrete_depth_sum.shape[1] + 1), dtype=torch.double).to(device)], axis=-1)
        for bin_idx_iter in range(bin_idx_max+1):
            cur_xy_tuples = torch.where(discrete_depths[img_i][0] == bin_idx_iter)
            if len(cur_xy_tuples[0]) == 0:
                continue
            discrete_depth_sum[:, bin_idx_iter] += (fmaps[img_i, :, cur_xy_tuples[0], cur_xy_tuples[1]]).sum(-1)

    return discrete_depth_sum


def compute_selectivity(depth_avg_response, unit_max_bin):
    """

    :param depth_avg_response: 2-dims: (unit_i, bin_i)
    :param unit_max_bin: 1-dim: unit_i
    :return: selectivity_index
    """
    if isinstance(depth_avg_response, np.ndarray):
        depth_avg_response = torch.from_numpy(depth_avg_response)
    response_max = depth_avg_response[np.arange(len(depth_avg_response)), unit_max_bin]
    response_nonmax = (depth_avg_response.sum(1) - response_max) / (depth_avg_response.shape[1] - 1)
    selectivity_index = (response_max - response_nonmax) / (response_max + response_nonmax + 1e-15)
    return selectivity_index