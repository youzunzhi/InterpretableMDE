import os
import argparse
import torch
import tqdm
from yacs.config import CfgNode as CN
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

from data import NYUv2_Dataloader, KITTI_Dataloader
from model import MFF_Model, BTS_Model
from utils import get_num_units_bins_and_nums_for_plt, get_sid_thresholds, discretize_depths, count_discrete_depth, \
    add_fmaps_on_discrete_depths, compute_selectivity


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = CN()
    cfg.MODEL_WEIGHTS_FILE = "model_weights/mff_senet_asn"
    cfg.LAYERS = 'D_MFF'
    cfg.ON_TRAINING_DATA = False

    # ---------------
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("opts", help="Modify configs using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg.merge_from_list(args.opts)
    # ---------------
    cfg.LAYERS = cfg.LAYERS.split('_')

    if 'mff_resnet' in cfg.MODEL_WEIGHTS_FILE:
        cfg.MODEL_NAME = 'MFF_resnet'
        cfg.DATASET = 'nyuv2'
    elif 'mff_senet' in cfg.MODEL_WEIGHTS_FILE:
        cfg.MODEL_NAME = 'MFF_senet'
        cfg.DATASET = 'nyuv2'
    elif 'bts_nyu' in cfg.MODEL_WEIGHTS_FILE:
        cfg.MODEL_NAME = 'BTS_nyu'
        cfg.DATASET = 'nyuv2'
    elif 'bts_kitti' in cfg.MODEL_WEIGHTS_FILE:
        cfg.MODEL_NAME = 'BTS_kitti'
        cfg.DATASET = 'kitti'
    else:
        raise NotImplementedError

    if cfg.DATASET == 'nyuv2':
        if cfg.ON_TRAINING_DATA:
            if 'BTS' in cfg.MODEL_NAME:
                cfg.DATASET_FILE = 'dataset/nyuv2/bts_train/bts_nyuv2_train.csv'
            else:
                cfg.DATASET_FILE = 'dataset/nyuv2/nyuv2_train.csv'
        else:
            cfg.DATASET_FILE = 'dataset/nyuv2/nyuv2_test.csv'
        cfg.MAX_DEPTH_EVAL = 10.
    elif cfg.DATASET == 'kitti':
        if cfg.ON_TRAINING_DATA:
            cfg.DATASET_FILE = 'dataset/kitti/eigen_train_files_with_gt.txt'
        else:
            cfg.DATASET_FILE = 'dataset/kitti/eigen_test_files_with_gt_652.txt'
        cfg.MAX_DEPTH_EVAL = 80.
    else:
        raise NotImplementedError
    # cfg.OUTPUT_DIR = f'outputs/dissect/bts_nyu_upconv2_eval'
    cfg.OUTPUT_DIR = f'outputs/dissect/{cfg.MODEL_WEIGHTS_FILE.split("/")[-1]}_{"train" if cfg.ON_TRAINING_DATA else "eval"}-[{(datetime.datetime.now()).strftime("%m%d%H%M%S")}]'
    print(cfg)

    if cfg.DATASET == 'nyuv2':
        dataloader = NYUv2_Dataloader('dissect', cfg.DATASET_FILE, cfg.MODEL_NAME, 1)
    elif cfg.DATASET == 'kitti':
        dataloader = KITTI_Dataloader('dissect', cfg.DATASET_FILE, cfg.MODEL_NAME, 1)
    else:
        raise NotImplementedError

    if 'MFF' in cfg.MODEL_NAME:
        model = MFF_Model(cfg.MODEL_NAME, cfg.MODEL_WEIGHTS_FILE, device).to(device)
    elif 'BTS' in cfg.MODEL_NAME:
        model = BTS_Model(cfg.DATASET, cfg.MAX_DEPTH_EVAL, cfg.MODEL_WEIGHTS_FILE).to(device)
    else:
        raise NotImplementedError

    dissect(model, cfg.MODEL_NAME, dataloader, cfg.DATASET, cfg.LAYERS, cfg.OUTPUT_DIR, device)


def dissect(model, model_name, dataloader, dataset_name, layers, output_dir, device):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    model.eval()
    num_units_list, num_bins_list, plt_row_list, plt_col_list, figsize_w_list, figsize_h_list = get_num_units_bins_and_nums_for_plt(layers, model_name)
    depth_cnt_list, depth_sum_list, depth_avg_list = [], [], []
    for layer_i in range(len(layers)):
        depth_cnt_list.append(torch.zeros(num_bins_list[layer_i], dtype=torch.double).to(device))
        depth_sum_list.append(torch.zeros((num_units_list[layer_i], num_bins_list[layer_i]), dtype=torch.double).to(device))
    sid_thresholds_list = [get_sid_thresholds(num_bins_list[layer_i], dataset_name) for layer_i in range(len(layers))]

    # -------------------- start batch iteration --------------------
    for batch in tqdm.tqdm(dataloader, desc='Dissection'):
        imgs = batch['image'].to(device)
        if 'MFF' in model_name:
            features_list = model.get_features(imgs, layers, also_get_output=False)
        elif 'BTS' in model_name:
            focals = batch['focal'].to(device) if 'focal' in batch.keys() else None
            features_list = model.get_features(imgs, layers, focals)
        else:
            raise NotImplementedError
        for layer_i in range(len(layers)):
            depths = batch['depth']
            discrete_concepts = discretize_depths(depths, dataset_name, thresholds=sid_thresholds_list[layer_i])
            depth_cnt_list[layer_i] = count_discrete_depth(discrete_concepts, depth_cnt_list[layer_i])
            depth_sum_list[layer_i] = add_fmaps_on_discrete_depths(discrete_concepts, features_list[layer_i],
                                                                   depth_sum_list[layer_i])
    # --------------- end of batch iteration --------------------

    for layer_i in range(len(layers)):
        depth_cnt_list[layer_i] = depth_cnt_list[layer_i].cpu().numpy()
        depth_sum_list[layer_i] = depth_sum_list[layer_i].cpu().numpy()
        depth_avg_list.append(depth_sum_list[layer_i] / (depth_cnt_list[layer_i] + 1e-15))

    unit_max_bin_list, selectivity_index_list = [], []
    for layer_i in range(len(layers)):
        depth_avg = depth_avg_list[layer_i]
        # ---- compute selectivity ----
        depth_avg_abs = abs(depth_avg)
        unit_max_bin = depth_avg_abs.argmax(-1)
        selectivity_index = compute_selectivity(depth_avg_abs, unit_max_bin).numpy()
        unit_max_bin_list.append(unit_max_bin)
        selectivity_index_list.append(selectivity_index)
        # --------------------------

        # ---- plot ----
        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(plt_row_list[layer_i], plt_col_list[layer_i],
                               figsize=(figsize_w_list[layer_i], figsize_h_list[layer_i]))

        for i in range(plt_row_list[layer_i]):
            for j in range(plt_col_list[layer_i]):
                unit_i = i * plt_col_list[layer_i] + j
                ax[i, j].set_title(
                    f'Unit {unit_i}, Layer {layers[layer_i]}, Selectivity: {selectivity_index[unit_i]:.04f}',
                    fontsize=12)
                ax[i, j].bar(range(len(depth_cnt_list[layer_i])), depth_avg[unit_i], color='red', width=0.85)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.subplots_adjust(wspace=0.15)
        selec_mean = selectivity_index.mean()
        save_fig_name = os.path.join(output_dir,
                                     f'{layers[layer_i]}-{selec_mean:.4f}.png')
        fig.savefig(save_fig_name, bbox_inches='tight')
        plt.close(fig)
        print(f'selec_mean_{layers[layer_i]}', selectivity_index.mean())

    return unit_max_bin_list, selectivity_index_list


if __name__ == '__main__':
    main()