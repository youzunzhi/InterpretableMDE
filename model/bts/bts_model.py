import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import encoder, bts
from ..utils import load_weights
from utils import log_info


class Model(nn.Module):
    def __init__(self, dataset, max_depth, model_weights_file, seed=None):
        super(Model, self).__init__()
        if model_weights_file != 'scratch':
            assert os.path.exists(model_weights_file)
            from_scratch = False
        else:
            from_scratch = True
        if seed is not None:
            torch.manual_seed(seed)

        self.encoder = encoder()
        self.decoder = bts(self.encoder.feat_out_channels, dataset, max_depth)
        self.dataset = dataset
        if not from_scratch:
            load_weights(self, model_weights_file)
        else:
            log_info('START FROM SCRATCH')

    def forward(self, x, focal=None):
        if focal is None:
            assert self.dataset == 'nyuv2'
            focal = 518.8579
        skip_feat = self.encoder(x)
        depth_8x8_scaled, depth_4x4_scaled, depth_2x2_scaled, reduc1x1, final_depth = self.decoder(skip_feat, focal)
        return final_depth

    def get_features(self, x, layer_list, focal=None, is_training=False):
        return_list = []
        if self.dataset == 'nyuv2':
            focal = 518.8579

        def forward(x, layer_list):
            skip_feat = self.encoder(x)
            skip0, skip1, skip2, skip3 = skip_feat[1], skip_feat[2], skip_feat[3], skip_feat[4]  # 64, 256, 512, 1024 | H/2, H/4, H/8, H/16
            dense_features = torch.nn.ReLU()(skip_feat[5])  # B, 2048, H/32, W/32
            upconv5 = self.decoder.upconv5(dense_features)  # B, 512, H/16, W/16
            upconv5 = self.decoder.bn5(upconv5)
            concat5 = torch.cat([upconv5, skip3], dim=1)  # B, 1536, H/16, W/16
            iconv5 = self.decoder.conv5(concat5)  # B, 512, H/16, W/16

            upconv4 = self.decoder.upconv4(iconv5)  # H/8
            upconv4 = self.decoder.bn4(upconv4)  # B, 256, H/8, W/8
            concat4 = torch.cat([upconv4, skip2], dim=1)  # B, 768, H/8, W/8
            iconv4 = self.decoder.conv4(concat4)
            iconv4 = self.decoder.bn4_2(iconv4)  # B, 256, H/8, W/8

            daspp_3 = self.decoder.daspp_3(iconv4)  # B, 128, H/8, W/8
            concat4_2 = torch.cat([concat4, daspp_3], dim=1)  # B, 896, H/8, W/8
            daspp_6 = self.decoder.daspp_6(concat4_2)  # B, 128, H/8, W/8
            concat4_3 = torch.cat([concat4_2, daspp_6], dim=1)  # B, 1024, H/8, W/8
            daspp_12 = self.decoder.daspp_12(concat4_3)  # B, 128, H/8, W/8
            concat4_4 = torch.cat([concat4_3, daspp_12], dim=1)  # B, 1152, H/8, W/8
            daspp_18 = self.decoder.daspp_18(concat4_4)  # B, 128, H/8, W/8
            concat4_5 = torch.cat([concat4_4, daspp_18], dim=1)
            daspp_24 = self.decoder.daspp_24(concat4_5)
            concat4_daspp = torch.cat([iconv4, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24],
                                      dim=1)  # B, 896, H/8, W/8
            daspp_feat = self.decoder.daspp_conv(concat4_daspp)  # B, 128, H/8, W/8

            reduc8x8 = self.decoder.reduc8x8(daspp_feat)  # B, 4, H/8, W/8
            plane_normal_8x8 = reduc8x8[:, :3, :, :]  # B, 3, H/8, W/8
            plane_normal_8x8 = F.normalize(plane_normal_8x8, 2, 1)
            plane_dist_8x8 = reduc8x8[:, 3, :, :]
            plane_eq_8x8 = torch.cat([plane_normal_8x8, plane_dist_8x8.unsqueeze(1)], 1)  # B, 4, H/8, W/8
            depth_8x8 = self.decoder.lpg8x8(plane_eq_8x8, focal)  # B, H, W
            depth_8x8_scaled = depth_8x8.unsqueeze(1) / self.decoder.max_depth
            depth_8x8_scaled_ds = F.interpolate(depth_8x8_scaled, scale_factor=0.25,
                                                            mode='nearest')  # B, 1, H/4, W/4

            upconv3 = self.decoder.upconv3(daspp_feat)  # H/4
            upconv3 = self.decoder.bn3(upconv3)  # B, 128, H/4, W/4
            concat3 = torch.cat([upconv3, skip1, depth_8x8_scaled_ds], dim=1)
            iconv3 = self.decoder.conv3(concat3)  # B, 128, H/4, W/4

            reduc4x4 = self.decoder.reduc4x4(iconv3)  # B, 4, H/4, W/4
            plane_normal_4x4 = reduc4x4[:, :3, :, :]
            plane_normal_4x4 = F.normalize(plane_normal_4x4, 2, 1)
            plane_dist_4x4 = reduc4x4[:, 3, :, :]
            plane_eq_4x4 = torch.cat([plane_normal_4x4, plane_dist_4x4.unsqueeze(1)], 1)
            depth_4x4 = self.decoder.lpg4x4(plane_eq_4x4, focal)
            depth_4x4_scaled = depth_4x4.unsqueeze(1) / self.decoder.max_depth
            depth_4x4_scaled_ds = F.interpolate(depth_4x4_scaled, scale_factor=0.5, mode='nearest')

            upconv2 = self.decoder.upconv2(iconv3)  # H/2
            upconv2 = self.decoder.bn2(upconv2)
            concat2 = torch.cat([upconv2, skip0, depth_4x4_scaled_ds], dim=1)
            iconv2 = self.decoder.conv2(concat2)  # B, 64, H/2, W/2

            reduc2x2 = self.decoder.reduc2x2(iconv2)  # B, 4, H/2, W/2
            plane_normal_2x2 = reduc2x2[:, :3, :, :]
            plane_normal_2x2 = F.normalize(plane_normal_2x2, 2, 1)
            plane_dist_2x2 = reduc2x2[:, 3, :, :]
            plane_eq_2x2 = torch.cat([plane_normal_2x2, plane_dist_2x2.unsqueeze(1)], 1)
            depth_2x2 = self.decoder.lpg2x2(plane_eq_2x2, focal)
            depth_2x2_scaled = depth_2x2.unsqueeze(1) / self.decoder.max_depth

            upconv1 = self.decoder.upconv1(iconv2)  # B, 32, H, W
            reduc1x1 = self.decoder.reduc1x1(upconv1)  # B, 1, H, W
            concat1 = torch.cat([upconv1, reduc1x1, depth_2x2_scaled, depth_4x4_scaled, depth_8x8_scaled],
                                dim=1)  # B, 36, H, W
            iconv1 = self.decoder.conv1(concat1)  # B, 32, H, W
            final_depth = self.decoder.max_depth * self.decoder.get_depth(iconv1)
            if self.dataset == 'kitti':
                final_depth = final_depth * focal.view(-1, 1, 1, 1).float() / 715.0873
            for layer in layer_list:
                if layer == 'upconv3':
                    return_list.append(upconv3)
                elif layer == 'iconv2':
                    return_list.append(iconv2)
                elif layer == 'upconv2':
                    return_list.append(upconv2)
                elif layer == 'iconv1':
                    return_list.append(iconv1)
                elif layer == 'upconv1':
                    return_list.append(upconv1)
                else:
                    raise NotImplementedError
            return_list.append(final_depth)
            return return_list

        if is_training:
            return_list = forward(x, layer_list)
        else:
            with torch.no_grad():
                return_list = forward(x, layer_list)
        return return_list

    def get_baseline_loss(self, preds, depths):
        if self.dataset == 'kitti':
            mask = depths > 1.0
        elif self.dataset == 'nyuv2':
            mask = depths > 0.1
        else:
            raise NotImplementedError
        variance_focus = 0.85
        silog_criterion = silog_loss(variance_focus=variance_focus)
        loss = silog_criterion.forward(preds, depths, mask.to(torch.bool))
        return loss


class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


