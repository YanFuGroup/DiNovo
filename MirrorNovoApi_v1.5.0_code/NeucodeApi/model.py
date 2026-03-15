
import time
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import config
from enum import Enum

import time

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

massAA_np = config.mass_AA_np
massAA_np_half = config.mass_AA_np_charge2


class TNet(nn.Module):
    """
    the T-net structure in the Point Net paper
    """

    def __init__(self, args):
        super(TNet, self).__init__()
        self.args = args
        self.num_units = args.units
        self.conv1 = nn.Conv1d(args.n_classes * args.num_ions + 1, self.num_units, 1)
        # self.conv1 = nn.Conv1d(args.n_classes * args.num_ions, self.num_units, 1)
        self.conv2 = nn.Conv1d(self.num_units, 2 * self.num_units, 1)
        self.conv3 = nn.Conv1d(2 * self.num_units, 4 * self.num_units, 1)
        self.fc1 = nn.Linear(4 * self.num_units, 2 * self.num_units)
        self.fc2 = nn.Linear(2 * self.num_units, self.num_units)

        self.output_layer = nn.Linear(self.num_units, args.n_classes)
        self.relu = nn.ReLU()

        self.input_batch_norm = nn.BatchNorm1d(args.n_classes * args.num_ions + 1)
        # self.input_batch_norm = nn.BatchNorm1d(args.n_classes * args.num_ions)

        self.bn1 = nn.BatchNorm1d(self.num_units)
        self.bn2 = nn.BatchNorm1d(2 * self.num_units)
        self.bn3 = nn.BatchNorm1d(4 * self.num_units)
        self.bn4 = nn.BatchNorm1d(2 * self.num_units)
        self.bn5 = nn.BatchNorm1d(self.num_units)

    def forward(self, x):
        """
        :param x: [batch * T, 26*8+1, N]
        :return:
            logit: [batch * T, 26]
        """
        x = self.input_batch_norm(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x, _ = torch.max(x, dim=2)  # global max pooling
        assert x.size(1) == 4 * self.num_units
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.output_layer(x)  # [batch * T, 26]
        return x


class Bulid_FEATURE(nn.Module):
    def __init__(self, distance_scale_factor, min_inten=1e-5):
        super(Bulid_FEATURE, self).__init__()
        self.min_inten = min_inten
        self.distance_scale_factor = distance_scale_factor

    # 1 try y  0 lys b
    def forward(self, location_index, peaks_location, peaks_intensity, ion_type):
        # 构建批次特征
        N = peaks_location.size(1)
        assert N == peaks_intensity.size(1)
        batch_size, T, vocab_size, num_ion = location_index.size()

        # mz_batches = torch.split(peaks_location, 1, dim=0)
        # inten_batches = torch.split(peaks_intensity, 1, dim=0)
        # result_list = [self.detect.detect(x[0].tolist(), y[0].tolist()) for x, y in zip(mz_batches, inten_batches)]

        # for mz in mz_batches:
        #     t = mz[0] * torch.tensor(result_list[0])

        # 确定了离子类型得峰
        # peaks_location2 = peaks_location * result_list
        # 删除确定了离子类型得峰
        # peaks_location = peaks_location * (1 - result_list)
        # 大小转换
        peaks_location = peaks_location.view(batch_size, 1, N, 1)
        # peaks_location2 = peaks_location2.view(batch_size, 1, N, 1)
        peaks_intensity = peaks_intensity.view(batch_size, 1, N, 1)

        peaks_location_mask = (peaks_location > self.min_inten).float()
        # peaks_location_mask2 = (peaks_location2 > self.min_inten).float()
        peaks_intensity = peaks_intensity.expand(-1, T, -1, -1)  # [batch, T, N, 1]

        location_index = location_index.view(batch_size, T, 1, vocab_size * num_ion)
        # location_index2 = location_index2.view(batch_size, T, 1, vocab_size * num_ion)

        location_index_mask = (location_index > self.min_inten).float()
        # location_index_mask2 = (location_index2 > self.min_inten).float()

        devation = torch.abs(peaks_location - location_index)
        # devation2 = torch.abs(peaks_location2 - location_index2)

        location_exp_minus_abs_diff = torch.exp(-devation * self.distance_scale_factor)
        # location_exp_minus_abs_diff2 = torch.exp(-devation2 * self.distance_scale_factor)
        # [batch, T, N, 26*8]
        location_exp_minus_abs_diff = location_exp_minus_abs_diff * peaks_location_mask * location_index_mask
        # location_exp_minus_abs_diff2 = location_exp_minus_abs_diff2 * peaks_location_mask2 * location_index_mask2
        # location_exp_minus_abs_diff = location_exp_minus_abs_diff * location_exp_minus_abs_diff2
        # 特征矩阵1：deviation拼接intensity

        input_feature = torch.cat((location_exp_minus_abs_diff, peaks_intensity), dim=3)

        return input_feature


class MirrorNovoMirrorNet(nn.Module):
    def __init__(self, args):
        super(MirrorNovoMirrorNet, self).__init__()
        self.args = args
        self.distance_scale_factor = config.distance_scale_factor
        self.build_node_feature = Bulid_FEATURE(self.distance_scale_factor)
        self.t_net = TNet(args=self.args)

    def forward(self, location_index, peaks_location, peaks_intensity, lys_location_index, lys_peaks_location,
                lys_peaks_intensity):
        try_input_feature = self.build_node_feature(location_index, peaks_location, peaks_intensity, 1)
        lys_input_feature = self.build_node_feature(lys_location_index, lys_peaks_location, lys_peaks_intensity, 0)
        input_feature = torch.cat((try_input_feature, lys_input_feature), dim=2)

        batch_size, T, N, in_features = input_feature.size()
        input_feature = input_feature.view(batch_size * T, N, self.args.n_classes * self.args.num_ions + 1)
        # input_feature = input_feature.view(batch_size * T, N, self.args.n_classes * self.args.num_ions)
        input_feature = input_feature.transpose(1, 2)
        result = self.t_net(input_feature).view(batch_size, T, self.args.n_classes)
        return result


MirrorNovo_Model = MirrorNovoMirrorNet


class Direction(Enum):
    forward = 1
    backward = 2


class InferenceModelWrapper(object):
    def __init__(self, forward_model: MirrorNovo_Model, backward_model: MirrorNovo_Model):
        self.forward_model = forward_model
        self.backward_model = backward_model
        self.forward_model.eval()
        self.backward_model.eval()

    def step(self, candidate_location, peaks_location, peaks_intensity,
             mirror_candidate_location, mirror_peaks_location, mirror_peaks_intensity,direction):
        """
        :param candidate_location: [batch, 1, 26, 8]
        :param peaks_location: [batch, N]
        :param peaks_intensity: [batch, N]
        """
        if direction == Direction.forward:
            model = self.forward_model
        else:
            model = self.backward_model
        # adj = self.bulid_adj(peaks_location)
        with torch.no_grad():
            logit = model(candidate_location, peaks_location, peaks_intensity, mirror_candidate_location,
                          mirror_peaks_location, mirror_peaks_intensity)
            logit = torch.squeeze(logit, dim=1)
            log_prob = F.log_softmax(logit, dim=1)
        return log_prob
