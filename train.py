import os
import argparse
import torch
import traceback
import datetime
from yacs.config import CfgNode as CN
from torch.utils.tensorboard import SummaryWriter

from data import NYUv2_Dataloader, KITTI_Dataloader
from model import MFF_Model, BTS_Model
from utils import log_info, setup_logger, get_num_units_bins_and_nums_for_plt, get_sid_thresholds, \
    discretize_depths, count_discrete_depth, add_fmaps_on_discrete_depths, compute_selectivity
from eval import evaluate
from dissect import dissect


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = CN()
    cfg.BATCH_SIZE = 16
    cfg.MODEL_NAME = 'MFF_resnet'   # [MFF_resnet|MFF_senet|BTS_nyu|BTS_kitti]

    cfg.LAYERS = 'D_MFF'
    cfg.MODEL_WEIGHTS_FILE = 'scratch'
    cfg.START_EPOCH = 1
    cfg.TOTAL_EPOCHS = 20
    cfg.LR = 1e-4
    cfg.ALPHA = -0.1

    cfg.SEED = 0
    # ---------
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("opts", help="Modify configs using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg.merge_from_list(args.opts)
    # ---------
    cfg.LAYERS = cfg.LAYERS.split('_')

    if 'MFF' in cfg.MODEL_NAME or 'nyu' in cfg.MODEL_NAME:
        cfg.DATASET = 'nyuv2'
    elif 'BTS_kitti' in cfg.MODEL_NAME:
        cfg.DATASET = 'kitti'
    else:
        raise NotImplementedError

    if cfg.DATASET == 'nyuv2':
        cfg.EVAL_DATASET_FILE = 'dataset/nyuv2/nyuv2_test.csv'
        cfg.DISSECT_DATASET_FILE = 'dataset/nyuv2/nyuv2_test.csv'
        cfg.MAX_DEPTH_EVAL = 10.
        if 'MFF' in cfg.MODEL_NAME:
            cfg.TRAIN_DATASET_FILE = 'dataset/nyuv2/nyuv2_train.csv'
        elif 'BTS' in cfg.MODEL_NAME:
            cfg.TRAIN_DATASET_FILE = 'dataset/nyuv2/bts_train/nyudepthv2_train_files_with_gt.txt'
            cfg.BATCH_SIZE = 4
            cfg.TOTAL_EPOCHS = 30
        else:
            raise NotImplementedError
    elif cfg.DATASET == 'kitti':
        cfg.EVAL_DATASET_FILE = 'data/kitti_data/eigen_test_files_with_gt_652.txt'
        cfg.DISSECT_DATASET_FILE = 'data/kitti_data/eigen_test_files_with_gt_652.txt'
        cfg.MAX_DEPTH_EVAL = 80.
        if 'BTS' in cfg.MODEL_NAME:
            cfg.TRAIN_DATASET_FILE = 'data/kitti_data/eigen_train_files_with_gt.txt'
            if cfg.MODEL_WEIGHTS_FILE == 'pretrained':
                cfg.LR = 1e-5
                if 'resnet' in cfg.MODEL_NAME:
                    cfg.MODEL_WEIGHTS_FILE = f'pretrained_model/BTS/bts_nyu_v2_pytorch_resnet50/model'
                elif 'densenet' in cfg.MODEL_NAME:
                    cfg.MODEL_WEIGHTS_FILE = f'pretrained_model/BTS/bts_nyu_v2_pytorch_densenet161/model'
                else:
                    raise NotImplementedError
            elif cfg.MODEL_WEIGHTS_FILE == 'scratch':
                cfg.LR = 1e-4
            else:
                raise NotImplementedError
    else:
        raise NotImplementedError

    cfg.OUTPUT_DIR = f'outputs/train/[{(datetime.datetime.now()).strftime("%m%d%H%M%S")}]'

    setup_logger(cfg.OUTPUT_DIR)
    log_info(cfg)

    # ---------------------
    if cfg.DATASET == 'nyuv2':
        dataloader = NYUv2_Dataloader
    elif cfg.DATASET == 'kitti':
        dataloader = KITTI_Dataloader
    else:
        raise NotImplementedError

    train_dataloader = dataloader('train', cfg.TRAIN_DATASET_FILE, cfg.MODEL_NAME, cfg.BATCH_SIZE, seed=cfg.SEED)
    eval_dataloader = dataloader('eval', cfg.EVAL_DATASET_FILE, cfg.MODEL_NAME, 1)
    dissect_dataloader = dataloader('dissect', cfg.DISSECT_DATASET_FILE, cfg.MODEL_NAME, 1)


    if 'MFF' in cfg.MODEL_NAME:
        model = MFF_Model(cfg.MODEL_NAME, cfg.MODEL_WEIGHTS_FILE, device, cfg.SEED).to(device)
        optimizer = torch.optim.Adam(model.parameters(), cfg.LR, weight_decay=1e-4)
    elif 'BTS' in cfg.MODEL_NAME:
        model = BTS_Model(cfg.MODEL_NAME, cfg.MODEL_WEIGHTS_FILE, cfg.DATASET, cfg.SEED).to(device)
        optimizer = torch.optim.AdamW([{'params': model.encoder.parameters(), 'weight_decay': 1e-2},
                                       {'params': model.decoder.parameters(), 'weight_decay': 0}],
                                      lr=cfg.LR, eps=1e-6)
        steps_per_epoch = len(train_dataloader)
        end_learning_rate = 0.1 * cfg.LR
        log_info(f'steps_per_epoch: {steps_per_epoch}')
        num_total_steps = cfg.TOTAL_EPOCHS * steps_per_epoch
        global_step = (cfg.START_EPOCH - 1) * steps_per_epoch
    else:
        raise NotImplementedError
    model = model.to(device)
    writer = SummaryWriter(cfg.OUTPUT_DIR)

    sid_thresholds_list = []
    num_bins_list = get_num_units_bins_and_nums_for_plt(cfg.LAYERS, cfg.MODEL_NAME)[1]
    for layer_i in range(len(cfg.LAYERS)):
        sid_thresholds_list.append(get_sid_thresholds(num_bins_list[layer_i], cfg.DATASET))

    for epoch in range(cfg.START_EPOCH, cfg.TOTAL_EPOCHS + 1):
        model.train()
        loss_sum, selectivity_sum, baseline_loss_sum = 0, 0, 0
        for batch_i, batch in enumerate(train_dataloader):
            imgs = batch['image'].to(device)
            depths = batch['depth'].to(device)
            if 'focal' in batch.keys():
                focals = batch['focal'].to(device)
            else:
                focals = None
            baseline_loss, selectivity = get_loss_with_regularizer(cfg.LAYERS, model, imgs, depths, sid_thresholds_list,
                                                                   cfg.DATASET, device, focals)
            loss = baseline_loss + cfg.ALPHA * selectivity
            optimizer.zero_grad()
            loss.backward()

            # ------ adjust LR ------
            if 'MFF' in cfg.MODEL_NAME:
                lr = cfg.LR * (0.1 ** (epoch // 5))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            elif 'BTS' in cfg.MODEL_NAME:
                lr = (cfg.LR - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                global_step += 1
            lr = optimizer.param_groups[0]['lr']

            optimizer.step()
            log_info(f"Epoch {epoch}/{cfg.TOTAL_EPOCHS}, Batch {batch_i}/{len(train_dataloader)}, LR {lr} "
                     f"baseline_loss {baseline_loss.data}, selectivity {selectivity}")
            loss_sum += loss.item()
            selectivity_sum += selectivity.item()
            baseline_loss_sum += baseline_loss.item()
            if batch_i % 100 == 0 and batch_i != 0:
                writer.add_scalar('baseline_loss', baseline_loss_sum/100, batch_i+(epoch-1)*len(train_dataloader))
                writer.add_scalar('selectivity', selectivity_sum/100, batch_i+(epoch-1)*len(train_dataloader))
                writer.add_scalar('loss', loss_sum/100, batch_i+(epoch-1)*len(train_dataloader))
                loss_sum, selectivity_sum, baseline_loss_sum = 0, 0, 0

        torch.save(model.state_dict(), 'model_saved.pth')

        # --- EVALUATION ---
        metrics = evaluate(model, cfg.MODEL_NAME, eval_dataloader, cfg.DATASET, cfg.MAX_DEPTH_EVAL, device)
        for metric_name, metric in metrics.items():
            writer.add_scalar(metric_name, metric.avg, epoch)

        # --- DISSECT ---
        unit_class_list, selectivity_index_list = dissect(model, cfg.MODEL_NAME, dissect_dataloader, cfg.DATASET,
                                                          cfg.LAYERS, os.path.join(cfg.OUTPUT_DIR, 'dissect'), device)
        for layer_i in range(len(cfg.LAYERS)):
            log_info(f'selec_idx_avg_{cfg.LAYERS[layer_i]}: {selectivity_index_list[layer_i].mean()}')
            writer.add_scalar(f'selec_idx_avg_{cfg.LAYERS[layer_i]}', selectivity_index_list[layer_i].mean(),
                              cfg.START_EPOCH - 1)


def get_loss_with_regularizer(layers, model, imgs, depths, sid_thresholds_list, dataset_name, device, focal=None):
    if focal is None:
        features_list = model.get_features(imgs, layers, also_get_output=True, is_training=True)
    else:
        features_list = model.get_features(imgs, layers, focal, also_get_output=True, is_training=True)

    preds = features_list[-1]
    baseline_loss = model.get_baseline_loss(preds, depths)

    # ---- selectivity regularizer ----
    selec_list = []
    for layer_i in range(len(layers)):
        discret_depths = discretize_depths(depths, dataset_name, thresholds=sid_thresholds_list[layer_i])
        concept_cnt = count_discrete_depth(discret_depths, torch.zeros(len(sid_thresholds_list[layer_i])-1, dtype=torch.double).to(device))
        features_list[layer_i] = torch.abs(features_list[layer_i])
        concept_sum = add_fmaps_on_discrete_depths(discret_depths, features_list[layer_i], torch.zeros((features_list[layer_i].shape[1], len(sid_thresholds_list[layer_i])-1), dtype=torch.double).to(device))
        concept_avg = concept_sum / (concept_cnt + 1e-15)

        num_units, num_bins = concept_avg.shape[0], concept_avg.shape[1]
        assert num_units >= num_bins
        unit_class = (torch.arange(num_units)//(num_units/num_bins)).to(concept_avg.device).long()
        layer_selec = compute_selectivity(concept_avg, unit_class)
        concept_cnt_iszero = (unit_class == torch.nonzero(concept_cnt==0)).sum(0, dtype=torch.bool)
        layer_selec[concept_cnt_iszero] = 0   # If d_k is absent in a batch, the unit k will be simply disregarded from the computation of the Loss.
        layer_selec = layer_selec.mean()
        selec_list.append(layer_selec)
    selectivity = sum(selec_list)
    return baseline_loss, selectivity


def adjust_learning_rate(init_lr, optimizer, epoch):
    lr = init_lr * (0.1 ** (epoch // 5))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        print(e)
        traceback.print_exc()
        input('Enter anything to quit screen')
