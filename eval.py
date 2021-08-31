import os
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from yacs.config import CfgNode as CN
import tqdm
import datetime
from terminaltables import AsciiTable

from data import NYUv2_Dataloader, KITTI_Dataloader
from model import MFF_Model, BTS_Model
from utils import AverageMeter, log_info, setup_logger


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = CN()
    cfg.MODEL_WEIGHTS_FILE = "model_weights/mff_senet_asn"
    cfg.OUTPUT_DIR = f'outputs/eval/[{(datetime.datetime.now()).strftime("%m%d%H%M%S")}]'

    # ---------------
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("opts", help="Modify configs using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg.merge_from_list(args.opts)
    # ---------------
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
        cfg.DATASET_FILE = 'dataset/nyuv2/nyuv2_test.csv'
        cfg.MAX_DEPTH_EVAL = 10.
    elif cfg.DATASET == 'kitti':
        cfg.DATASET_FILE = 'dataset/kitti/eigen_test_files_with_gt_652.txt'
        cfg.MAX_DEPTH_EVAL = 80.
    else:
        raise NotImplementedError

    print(cfg)
    # ------------------------------------------------------------------------------------------

    if cfg.DATASET == 'nyuv2':
        dataloader = NYUv2_Dataloader('eval', cfg.DATASET_FILE, cfg.MODEL_NAME, 1)
    elif cfg.DATASET == 'kitti':
        dataloader = KITTI_Dataloader('eval', cfg.DATASET_FILE, cfg.MODEL_NAME, 1)
    else:
        raise NotImplementedError

    if 'MFF' in cfg.MODEL_NAME:
        model = MFF_Model(cfg.MODEL_NAME, cfg.MODEL_WEIGHTS_FILE, device).to(device)
    elif 'BTS' in cfg.MODEL_NAME:
        model = BTS_Model(cfg.DATASET, cfg.MAX_DEPTH_EVAL, cfg.MODEL_WEIGHTS_FILE).to(device)
    else:
        raise NotImplementedError
    evaluate(model, cfg.MODEL_NAME, dataloader, cfg.DATASET, cfg.MAX_DEPTH_EVAL, device, output_dir=cfg.OUTPUT_DIR)


def evaluate(model, model_name, eval_dataloader, dataset_name, max_depth_eval, device, output_dir=None):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        setup_logger(output_dir)
    model.eval()
    metrics = make_metrics_dict()
    for batch in tqdm.tqdm(eval_dataloader, desc='EVALUATING'):
        imgs = batch['image'].to(device)
        depths = batch['depth']
        assert len(imgs) == 1
        with torch.no_grad():
            if 'MFF' in model_name:
                preds = model(imgs)
                preds = F.interpolate(preds, size=[228, 304], mode='bilinear', align_corners=False)
            elif 'BTS' in model_name:
                focals = batch['focal'].to(device) if 'focal' in batch.keys() else None
                preds = model(imgs, focals)
            else:
                raise NotImplementedError
        batch_metrics = get_batch_metrics(dataset_name, depths, preds, max_depth_eval)
        for metric_name, metric in metrics.items():
            metric.update(batch_metrics[metric_name].avg, n=batch_metrics[metric_name].count)

    log_str = f'[Test on {len(eval_dataloader.dataset)} images]\n'
    log_str += get_metrics_table(metrics)
    log_info(log_str)
    return metrics


def get_batch_metrics(dataset_name, gt_depth, pred_depth, max_depth_eval=None):
    gt_depth, pred_depth, _ = mask_before_eval(dataset_name, gt_depth, pred_depth, max_depth_eval)
    d102, d105, d110, a1, a2, a3, rmse, rmse_log, abs_rel, sq_rel, log10 = compute_metrics(gt_depth, pred_depth)

    batch_metrics = make_metrics_dict()
    batch_metrics['d102'].update(d102)
    batch_metrics['d105'].update(d105)
    batch_metrics['d110'].update(d110)
    batch_metrics['a1'].update(a1)
    batch_metrics['a2'].update(a2)
    batch_metrics['a3'].update(a3)
    batch_metrics['rmse'].update(rmse)
    batch_metrics['rmse_log'].update(rmse_log)
    batch_metrics['abs_rel'].update(abs_rel)
    batch_metrics['sq_rel'].update(sq_rel)
    batch_metrics['log10'].update(log10)
    return batch_metrics


def compute_metrics(gt_depth, pred_depth):
    def lg10(x):
        return np.log(x)/np.log(10)

    # accuracy
    threshold = np.maximum((gt_depth / pred_depth), (pred_depth / gt_depth))
    d102 = (threshold < 1.02).mean()
    d105 = (threshold < 1.05).mean()
    d110 = (threshold < 1.10).mean()
    a1 = (threshold < 1.25).mean()
    a2 = (threshold < 1.25 ** 2).mean()
    a3 = (threshold < 1.25 ** 3).mean()

    # MSE
    rmse = (gt_depth - pred_depth) ** 2
    rmse = np.sqrt(rmse.mean())

    # MSE(log)
    rmse_log = (np.log(gt_depth) - np.log(pred_depth)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    # Abs Relative difference
    abs_rel = np.mean(np.abs(gt_depth - pred_depth) / gt_depth)

    # Squared Relative difference
    sq_rel = np.mean(((gt_depth - pred_depth) ** 2) / gt_depth)

    # log 10
    log10 = np.mean(np.abs(lg10(gt_depth) - lg10(pred_depth)))

    return d102, d105, d110, a1, a2, a3, rmse, rmse_log, abs_rel, sq_rel, log10


def mask_before_eval(dataset_name, gt_depth, pred_depth, max_depth_eval, min_depth_eval=1e-3):
    if isinstance(gt_depth, torch.Tensor):
        gt_depth = gt_depth.cpu().numpy()
    if isinstance(pred_depth, torch.Tensor):
        pred_depth = pred_depth.cpu().numpy()
    if max_depth_eval is None:
        if dataset_name == 'nyuv2':
            max_depth_eval = 10.
        elif dataset_name == 'kitti':
            max_depth_eval = 80.

    pred_depth[pred_depth < min_depth_eval] = min_depth_eval
    pred_depth[pred_depth > max_depth_eval] = max_depth_eval
    pred_depth[np.isinf(pred_depth)] = max_depth_eval
    valid_mask = np.logical_and(gt_depth > min_depth_eval, gt_depth < max_depth_eval)

    if dataset_name == 'kitti':
        # do garg crop
        gt_height, gt_width = gt_depth.shape[2], gt_depth.shape[3]
        garg_crop_mask = np.zeros(valid_mask.shape)
        garg_crop_mask[:, :, int(0.40810811 * gt_height):int(0.99189189 * gt_height),
        int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

        valid_mask = np.logical_and(valid_mask, garg_crop_mask)

    return gt_depth[valid_mask], pred_depth[valid_mask], valid_mask


def make_metrics_dict():
    metrics = {
        'd102': AverageMeter(),
        'd105': AverageMeter(),
        'd110': AverageMeter(),
        'a1': AverageMeter(),
        'a2': AverageMeter(),
        'a3': AverageMeter(),
        'rmse': AverageMeter(),
        'rmse_log': AverageMeter(),
        'abs_rel': AverageMeter(),
        'sq_rel': AverageMeter(),
        'log10': AverageMeter()
    }
    return metrics


def get_metrics_table(metrics):
    metrics_table = [["Metrics", "Result"]]
    log_str_formats = {name: "%.6f" for name in metrics.keys()}
    for metric_name, metric in metrics.items():
        result = log_str_formats[metric_name] % metric.avg
        metrics_table += [[metric_name, result]]
    return AsciiTable(metrics_table).table


if __name__ == '__main__':
    main()