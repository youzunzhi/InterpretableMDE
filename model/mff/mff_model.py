import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet50
from .senet import senet154
from .modules import E_resnet, E_senet, D, MFF, R
from .sobel import Sobel
from ..utils import load_weights
from utils import log_info


class Model(nn.Module):
    def __init__(self, model_name, model_weights_file, device, seed=None):
        super(Model, self).__init__()
        if model_weights_file != 'scratch':
            assert os.path.exists(model_weights_file)
            from_scratch = False
        else:
            from_scratch = True
        if seed is not None:
            torch.manual_seed(seed)

        encoder = model_name.split('_')[-1]
        if encoder == 'resnet':
            if from_scratch:
                original_model = resnet50(pretrained=True)
            else:
                original_model = resnet50(pretrained=False)
            self.E = E_resnet(original_model)
            num_features = 2048
            block_channel = [256, 512, 1024, 2048]
        elif encoder == 'senet':
            if from_scratch:
                original_model = senet154(pretrained='imagenet')
            else:
                original_model = senet154(pretrained=None)
            self.E = E_senet(original_model)
            num_features = 2048
            block_channel = [256, 512, 1024, 2048]
        else:
            raise NotImplementedError

        self.D = D(num_features)
        self.MFF = MFF(block_channel)
        self.R = R(block_channel)

        if not from_scratch:
            load_weights(self, model_weights_file)
        else:
            log_info('START FROM SCRATCH')

        self.device = device

    def forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4, [x_decoder.size(2), x_decoder.size(3)])
        out = self.R(torch.cat((x_decoder, x_mff), 1))

        return out

    def get_features(self, x, layer_list, also_get_output=True, is_training=False):
        return_list = []

        def forward(x, layer_list, also_get_output):
            x_block1, x_block2, x_block3, x_block4 = self.E(x)
            x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
            x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4, [x_decoder.size(2), x_decoder.size(3)])
            for layer in layer_list:
                if layer == 'D':
                    return_list.append(x_decoder)
                elif layer == 'MFF':
                    return_list.append(x_mff)
                elif layer == 'Rconv0':
                    x_rconv0 = F.relu(self.R.bn0(self.R.conv0(torch.cat((x_decoder, x_mff), 1))))
                    return_list.append(x_rconv0)
                elif layer == 'Rconv1':
                    x_rconv0 = F.relu(self.R.bn0(self.R.conv0(torch.cat((x_decoder, x_mff), 1))))
                    x_rconv1 = F.relu(self.R.bn1(self.R.conv1(x_rconv0)))
                    return_list.append(x_rconv1)
                else:
                    raise NotImplementedError
            if also_get_output:
                if 'Rconv1' not in layer_list:
                    x_rconv0 = F.relu(self.R.bn0(self.R.conv0(torch.cat((x_decoder, x_mff), 1))))
                    x_rconv1 = F.relu(self.R.bn1(self.R.conv1(x_rconv0)))
                out = self.R.conv2(x_rconv1)
                return_list.append(out)
            return return_list

        if is_training:
            return_list = forward(x, layer_list, also_get_output)
        else:
            with torch.no_grad():
                return_list = forward(x, layer_list, also_get_output)
        return return_list

    def get_baseline_loss(self, outputs, depths):
        get_gradient = Sobel().to(self.device)
        cos = nn.CosineSimilarity(dim=1, eps=0)
        ones = torch.ones(depths.size(0), 1, depths.size(2), depths.size(3)).float().to(self.device)

        depths_grad = get_gradient(depths)
        output_grad = get_gradient(outputs)
        depth_grad_dx = depths_grad[:, 0, :, :].contiguous().view_as(depths)
        depth_grad_dy = depths_grad[:, 1, :, :].contiguous().view_as(depths)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depths)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depths)

        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

        # depth_normal = F.normalize(depth_normal, p=2, dim=1)
        # output_normal = F.normalize(output_normal, p=2, dim=1)

        loss_depth = torch.log(torch.abs(outputs - depths) + 0.5).mean()
        loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
        loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
        loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()

        loss = loss_depth + loss_normal + (loss_dx + loss_dy)
        return loss