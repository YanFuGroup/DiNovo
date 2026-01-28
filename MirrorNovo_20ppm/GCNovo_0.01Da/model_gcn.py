#!/anaconda3/python3.7
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: model_gcn.py
# @Author: MCN
# @E-mail: changning_m@163.com
# @Time:  2020/08/29
# ---
import time
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import config
import torch.nn.init as init
import torch.optim as optim
from enum import Enum
# from model_transformer import Transformer
import time


massAA_np = config.mass_AA_np
massAA_np_half = config.mass_AA_np_charge2

class GraphConvolution_Tnet(nn.Module):
    def __init__(self, input_dim, output_dim, units, use_bias=False):
        """图卷积：L*X*W

        args ：
        input_dim:节点特征维度
        output_dim:输出特征维度
        """
        super(GraphConvolution_Tnet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.units = units
        # self.conv = nn.Conv1d(self.input_dim, units, 1)
        # self.bn = nn.BatchNorm1d(units)


        self.use_bias = use_bias
        # self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        # self.reset_parameters()

    def reset_parameters(self):
        # init.kaiming_uniform_(self.weight)
        # if self.use_bias:
        #     init.zeros_(self.bias)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adjacency, input_feature):
        """adjacency:[batch,num_peaks,num_peaks]
            input_feature: [batch,pep_len_padding,num_num_peaks,feature]
            return :[batch,pep_len_padding,num_peak,output_dim]
        """
        # batch_size, T, N, in_features = input_feature.size()
        # # input_feature = input_feature.view(batch_size*T, N, in_features)
        # input_feature = input_feature.transpose(1, 2)
        #
        # # support = self.bn(self.conv(input_feature)).transpose(1, 2)
        # input_feature = input_feature.transpose(1, 2)
        # support = input_feature.view(batch_size, T, N, -1)

        # support = torch.matmul(input_feature, self.weight)
        support = torch.cat(torch.split(input_feature, 1, dim=1), dim=-1)
        support_dim = support.size()
        output = torch.matmul(adjacency, support.view(support_dim[0], support_dim[1] * support_dim[2], support_dim[3]))
        output = torch.stack(torch.split(output, self.output_dim, dim=-1), dim=1)

        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*W

        args ：
        input_dim:节点特征维度
        output_dim:输出特征维度
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # init.kaiming_uniform_(self.weight)
        # if self.use_bias:
        #     init.zeros_(self.bias)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adjacency, input_feature):
        """adjacency:[batch,num_peaks,num_peaks]
            input_feature: [batch,pep_len_padding,num_num_peaks,feature]
            return :[batch,pep_len_padding,num_peak,output_dim]
        """
        # print(input_feature.shape, self.weight.shape)
        support = torch.matmul(input_feature, self.weight)
        support = torch.cat(torch.split(support, 1, dim=1), dim=-1)
        support_dim = support.size()
        output = torch.matmul(adjacency, support.view(support_dim[0], support_dim[1] * support_dim[2], support_dim[3]))
        output = torch.stack(torch.split(output, self.output_dim, dim=-1), dim=1)

        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'

class MLP_layer(nn.Module):
    def __init__(self, args):
        super(MLP_layer, self).__init__()
        self.fc1 = nn.Linear(args.input_dim, args.units)
        self.fc2 = nn.Linear(args.units, 2 * args.units)
        self.fc3 = nn.Linear(2 * args.units, 4 * args.units)

        self.input_batch_norm = nn.BatchNorm1d(args.input_dim)
        self.bn1 = nn.BatchNorm1d(args.units)
        self.bn2 = nn.BatchNorm1d(2 * args.units)
        self.bn3 = nn.BatchNorm1d(4 * args.units)

    def forward(self, x):

        x = self.input_batch_norm(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))

        return x

class TNet(nn.Module):
    """
    the T-net structure in the Point Net paper
    """
    def __init__(self, args):
        super(TNet, self).__init__()
        self.conv1 = nn.Conv1d(args.input_dim, args.units, 1)
        self.conv2 = nn.Conv1d(args.units, 2 * args.units, 1)
        self.conv3 = nn.Conv1d(2 * args.units, 4 * args.units, 1)
        # self.conv4 = nn.Conv1d(4 * args.units, 8 * args.units, 1)
        # self.fc1 = nn.Linear(4 * args.num_units, 2 * args.num_units)
        # self.fc2 = nn.Linear(2*num_units, num_units)
        # if not with_lstm:
        #     self.output_layer = nn.Linear(num_units, config.vocab_size)
        # self.relu = nn.ReLU()

        self.input_batch_norm = nn.BatchNorm1d(args.input_dim)

        self.bn1 = nn.BatchNorm1d(args.units)
        self.bn2 = nn.BatchNorm1d(2 * args.units)
        self.bn3 = nn.BatchNorm1d(4 * args.units)
        # self.bn4 = nn.BatchNorm1d(8 * args.units)

    def forward(self, x):
        """
        :param x: [batch * T, 26*8+1, N]
        :return:
            logit: [batch * T, 26]
        """
        # print(x.shape)
        x = self.input_batch_norm(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  # x: [batch * T, 256, N]
        # x = F.relu(self.bn4(self.conv4(x)))
        # x, _ = torch.max(x, dim=2)  # global max pooling
        # assert x.size(1) == 4*num_units

        # x = activation_func(self.bn4(self.fc1(x)))
        # x = activation_func(self.bn5(self.fc2(x)))
        # if not self.with_lstm:
        #     x = self.output_layer(x)  # [batch * T, 26]
        return x

class Bulid_ADJ(nn.Module):
    def __init__(self, massAA_np, distance_scale_factor, args, flag=True,device=0):
        super(Bulid_ADJ, self).__init__()
        self.args = args
        self.distance_scale_factor = distance_scale_factor
        self.ONE_TENSOR = torch.tensor(1.0).to(device)
        self.ZERO_TENSOR = torch.tensor(0.0).to(device)
        self.EYE_TENSOR = torch.eye(self.args.MAX_NUM_PEAK).to(device)
        self.MASSAA_TENSOR = torch.from_numpy(massAA_np.reshape(1, len(massAA_np), 1, 1)).to(device)
        # self.MASSAA_TENSOR_2 = torch.from_numpy(massAA_np_half.reshape(1, len(massAA_np_half), 1, 1)).to(device)
        self.massAA_dim = self.MASSAA_TENSOR.size()
        self.edge_embedding = nn.Embedding(args.edges_classes + 2, args.units)
        self.device = device

    def forward(self, peaks_location, peaks_intensity, precursormass):
        batch_size, N = peaks_location.size()
        PRECURSORMASS_TENSOR = precursormass.view(batch_size, 1, 1)
        peaks_location1 = peaks_location.unsqueeze(1)
        peaks_intensity1 = peaks_intensity.unsqueeze(1)

        adj_mask1 = (peaks_intensity1 > 1e-5).float()
        adj_mask2 = (peaks_intensity1.permute(0, 2, 1) > 1e-5).float()

        peak_matrix = torch.abs(peaks_location1 - peaks_location1.permute(0, 2, 1))
        peak_matrix = peak_matrix * adj_mask1 * adj_mask2
        peak_matrix_by = peaks_location1 + peaks_location1.permute(0, 2, 1)
        peak_matrix_by = peak_matrix_by * adj_mask1 * adj_mask2
        adj_build = torch.unsqueeze(peak_matrix, 1)
        adj = torch.abs(adj_build - self.MASSAA_TENSOR)
        adj_build_by = torch.abs(peak_matrix_by - PRECURSORMASS_TENSOR).unsqueeze(1)
        adj = torch.cat([adj, adj_build_by], dim=1)

        min_value, min_indices = adj.min(dim=1)  # [b, N, N] 有可能有一条边，也有可能没有边
        min_indices = min_indices + torch.tensor([1]).to(self.device)  # 预留出来0作为没有边的embedding

        adj = (adj <= 0.02).float()
        adj = adj.sum(dim=1)  # [b, N, N]
        adj = (adj > 0.).float()  # [b, N, N]

        # add edge feature
        min_indices = torch.mul(min_indices, adj.long())  # [b, N, N]
        edge_embedding = self.edge_embedding(min_indices)   # [b, N, N, embedding]
        edge_embedding = torch.matmul(adj.unsqueeze(dim=2), edge_embedding).squeeze(dim=2)  # [b, N, embedding]

        adj = adj + self.EYE_TENSOR

        degree = torch.sum(adj, dim=2)
        # print(degree)
        # edge_embedding = edge_embedding / degree.unsqueeze(dim=-1)

        d_hat = torch.diag_embed(torch.pow(degree, -0.5))

        adj = torch.matmul(torch.matmul(d_hat, adj), d_hat)
        # return adj
        return adj, edge_embedding

class Centrality_Encoder(nn.Module):
    """
    Compute degree features for each node in the graph.
    """

    def __init__(self, num_degree, embedding_dim):
        super(Centrality_Encoder, self).__init__()
        self.num_degree = num_degree
        self.embedding_dim = embedding_dim

        self.degree_encoder = nn.Embedding(num_degree, embedding_dim, padding_idx=0)

    def forward(self, degree, x):
        """
        x.shape: [B, T, N , F_in]
        B: batch size; T: squence length; N: num nodes; F_in: node features dim
        degree.shape: [B, N]
        B: batch size; N: num nodes
        """
        T = x.size()[1]
        # print(self.degree_encoder(degree))
        x = x + self.degree_encoder(degree).unsqueeze(1).repeat(1, T, 1, 1)
        return x

class Bulid_FEATURE(nn.Module):
    def __init__(self, distance_scale_factor, min_inten=1e-5):
        super(Bulid_FEATURE, self).__init__()
        self.min_inten = min_inten
        self.distance_scale_factor = distance_scale_factor
    def forward(self, location_index, peaks_location, peaks_intensity):
        # 构建批次特征
        N = peaks_location.size(1)
        assert N == peaks_intensity.size(1)
        batch_size, T, vocab_size, num_ion = location_index.size()

        peaks_location = peaks_location.view(batch_size, 1, N, 1)
        peaks_intensity = peaks_intensity.view(batch_size, 1, N, 1)
        # peaks_location = peaks_location.expand(-1, T, -1, -1)  # [batch, T, N, 1]

        peaks_location_mask = (peaks_location > self.min_inten).float()
        peaks_intensity = peaks_intensity.expand(-1, T, -1, -1)  # [batch, T, N, 1]

        # location_index = location_index.sum(dim=3)
        location_index = location_index.view(batch_size, T, 1, vocab_size * num_ion)
        location_index_mask = (location_index > self.min_inten).float()

        location_exp_minus_abs_diff = torch.exp(
            -torch.abs(peaks_location - location_index) * self.distance_scale_factor)
        # [batch, T, N, 26*8]
        location_exp_minus_abs_diff = location_exp_minus_abs_diff * peaks_location_mask * location_index_mask

        input_feature = torch.cat((location_exp_minus_abs_diff, peaks_intensity), dim=3)

        # 根据谱峰的强度构建特征
        # peaks_intensity_1 = peaks_intensity.view(batch_size, N, 1)
        # peaks_intensity_2 = peaks_intensity.view(batch_size, 1, N)
        # peaks_intensity = peaks_intensity_1 + peaks_intensity_2
        # adj = torch.mul(adj, peaks_intensity)

        # return adj, location_exp_minus_abs_diff
        return input_feature

class InitNet(nn.Module):
    def __init__(self, args):
        super(InitNet, self).__init__()
        self.lstm_hidden_units = args.lstm_hidden_units
        self.num_lstm_layers = args.num_lstm_layers
        self.init_state_layer = nn.Linear(args.embedding_size, 2 * args.lstm_hidden_units)

    def forward(self, spectrum_representation):
        """

        :param spectrum_representation: [N, embedding_size]
        :return:
            [num_lstm_layers, batch_size, lstm_units], [num_lstm_layers, batch_size, lstm_units],
        """
        x = torch.tanh(self.init_state_layer(spectrum_representation))
        h_0, c_0 = torch.split(x, self.lstm_hidden_units, dim=1)
        h_0 = torch.unsqueeze(h_0, dim=0)
        h_0 = h_0.repeat(self.num_lstm_layers, 1, 1)
        c_0 = torch.unsqueeze(c_0, dim=0)
        c_0 = c_0.repeat(self.num_lstm_layers, 1, 1)
        return h_0, c_0

class GCNovo(nn.Module):
    def __init__(self, args,device):
        super(GCNovo, self).__init__()
        self.args = args
        self.input_dim = self.args.input_dim
        self.output_dim = self.args.output_dim

        self.distance_scale_factor = config.distance_scale_factor
        self.bulid_adj = Bulid_ADJ(massAA_np, distance_scale_factor=self.distance_scale_factor, args=self.args,device=device)
        self.build_node_feature = Bulid_FEATURE(self.distance_scale_factor)
        self.t_net = TNet(args=self.args)

        self.gcn1 = GraphConvolution(4 * args.units, 4 * args.units)
        self.gcn7 = GraphConvolution(4 * args.units, 4 * args.units)

        self.fc1 = nn.Linear(4 * args.units, 2 * args.units)
        self.fc2 = nn.Linear(2 * args.units, args.units)
        self.bn1 = nn.BatchNorm1d(2 * args.units)
        self.bn2 = nn.BatchNorm1d(args.units)

        self.output_layer = nn.Linear(args.units, self.args.n_classes)

        self.relu = nn.ReLU()

    def forward(self, location_index, peaks_location, peaks_intensity, precursormass, aa_input=None, state_tuple=None):
        """location_index:[batch,len_padding,26,12]
            peaks_location：[batch,max_peaks_num]
            peaks_intensity:[batch,max_peaks_num]
            return:logits:[batch,len_padding,26]
        # 构建图            """
        adj, edge_embedding = self.bulid_adj(peaks_location, peaks_intensity, precursormass)
        input_feature = self.build_node_feature(location_index, peaks_location, peaks_intensity)

        batch_size, T, N, in_features = input_feature.size()
        # print(input_feature.shape)
        edge_embedding = edge_embedding.unsqueeze(dim=1).repeat(1, T, 1, 1)  # [b, T, N, embedding]
        input_feature = torch.cat((input_feature, edge_embedding), dim=-1)
        input_feature = input_feature.view(batch_size*T, N, -1)
        input_feature = input_feature.transpose(1, 2)

        input_feature = self.t_net(input_feature)

        input_feature = input_feature.transpose(1, 2)
        input_feature = input_feature.view(batch_size, T, N, -1)

        x = self.relu(self.gcn1(adj, input_feature))
        x = self.relu(self.gcn7(adj, x))
        # print(x.shape)
        x = F.leaky_relu(x.sum(dim=2))  # b, t, 256
        x = x.view(batch_size*T, -1)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = x.view(batch_size, T, -1)
        logits = self.output_layer(x)

        return logits

class denovo_GCN(nn.Module):
    """定义两层的GraphConvolution模型"""

    def __init__(self, args):
        super(denovo_GCN, self).__init__()
        self.args = args
        self.input_dim = self.args.input_dim
        self.output_dim = self.args.output_dim

        self.distance_scale_factor = config.distance_scale_factor
        self.bulid_adj = Bulid_ADJ(massAA_np, distance_scale_factor=self.distance_scale_factor, args=self.args)
        self.build_node_feature = Bulid_FEATURE(self.distance_scale_factor)
        self.t_net = TNet(args=self.args)
        self.embedding = nn.Embedding(num_embeddings=args.n_classes, embedding_dim=args.embedding_size)

        # self.centrality_encoder = Centrality_Encoder(num_degree=64, embedding_dim=256)
        # self.mlp = MLP_layer(args=self.args)
        # self.gcn1 = GraphConvolution_Tnet(self.input_dim, self.output_dim, units=2*args.units)

        self.gcn1 = GraphConvolution(4 * args.units, 4 * args.units)
        self.gcn7 = GraphConvolution(4 * args.units, 4 * args.units)

        self.lstm = nn.LSTM(args.embedding_size,
                            args.lstm_hidden_units,
                            num_layers=args.num_lstm_layers,
                            batch_first=True)

        self.fc1 = nn.Linear(4 * args.units, 2 * args.units)
        self.fc2 = nn.Linear(2 * args.units, args.units)
        self.bn1 = nn.BatchNorm1d(2 * args.units)
        self.bn2 = nn.BatchNorm1d(args.units)

        # self.output_layer1 = nn.Linear(args.output_dim + args.embedding_size, 2 * args.units)
        # self.output_layer2 = nn.Linear(2 * args.units, args.units)
        self.output_layer3 = nn.Linear(args.units + args.embedding_size, self.args.n_classes)
        # self.output_layer3 = nn.Linear(2 * args.units, self.args.n_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, location_index, peaks_location, peaks_intensity, aa_input=None, state_tuple=None):
        """location_index:[batch,len_padding,26,12]
            peaks_location：[batch,max_peaks_num]
            peaks_intensity:[batch,max_peaks_num]
            return:logits:[batch,len_padding,26]
        # 构建图            """
        adj, edge_embedding = self.bulid_adj(peaks_location, peaks_intensity)
        # adj = self.bulid_adj(peaks_location, peaks_intensity)
        input_feature = self.build_node_feature(location_index, peaks_location, peaks_intensity)
        # adj, input_feature = self.build_node_feature(adj, location_index, peaks_location, peaks_intensity)
        # print(input_feature.shape)
        # peaks_location_mask = torch.where(peaks_location > 0, torch.tensor([1]).to(device), torch.tensor([0]).to(device))
        # peaks_intensity_mask = torch.where(peaks_intensity > 0, torch.tensor([1]).to(device),
        #                                   torch.tensor([0]).to(device))
        batch_size, T, N, in_features = input_feature.size()
        # print(input_feature.shape)

        # add edge feature
        edge_embedding = edge_embedding.unsqueeze(dim=1).repeat(1, T, 1, 1)  # [b, T, N, embedding]
        input_feature = torch.cat((input_feature, edge_embedding), dim=-1)


        input_feature = input_feature.view(batch_size*T, N, -1)
        input_feature = input_feature.transpose(1, 2)

        # print("input feature: ", input_feature.shape)
        input_feature = self.t_net(input_feature)
        # x = self.mlp(input_feature)

        input_feature = input_feature.transpose(1, 2)
        input_feature = input_feature.view(batch_size, T, N, -1)
        # print("peaks_intensity_mask.shape: ", peaks_intensity_mask.shape)
        # degree = degree.long() * peaks_location_mask * peaks_intensity_mask
        # print("degree: ", degree.shape, degree)
        # print(degree.max())
        # assert degree.max() <= 64, "Error: max degree > 512"
        # x = self.centrality_encoder(degree, x)
        # print("input feature: ", input_feature.shape)
        x = self.relu(self.gcn1(adj, input_feature))
        x = self.relu(self.gcn7(adj, x))
        # print(x.shape)

        x = F.leaky_relu(x.sum(dim=2))  # b, t, 256
        x = x.view(batch_size*T, -1)
        x = self.relu(self.bn1(self.fc1(x)))

        x = self.relu(self.bn2(self.fc2(x)))
        x = x.view(batch_size, T, -1)

        aa_embedded = self.embedding(aa_input)
        output_feature, new_state_tuple = self.lstm(aa_embedded, state_tuple)
        output_feature = torch.cat((x, self.relu(output_feature)), dim=2)
        output_feature = self.dropout(output_feature)

        # print(output_feature.shape)
        # output_feature = self.relu(self.output_layer1(output_feature))
        # x = self.relu(self.output_layer2(x))
        logits = self.output_layer3(output_feature)

        return logits, new_state_tuple

GCNovoModel = GCNovo


class Direction(Enum):
    forward = 1
    backward = 2


class InferenceModelWrapper(object):
    def __init__(self, forward_model: GCNovoModel, backward_model: GCNovoModel, init_net: InitNet=None):
        self.forward_model = forward_model
        self.backward_model = backward_model
        self.forward_model.eval()
        self.backward_model.eval()

        if False:
            assert init_net is not None
            self.init_net = init_net
            self.init_net.eval()

    def step(self, candidate_location, peaks_location, peaks_intensity, precursormass, aa_input, state_tuple, direction):
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
        # num_params = sum(param.numel() for param in model.parameters())
        # trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print("forward(M):", num_params, ",trainable:", trainable_num)
        if False:
            logit, new_state_tuple = model(candidate_location, peaks_location, peaks_intensity, precursormass, aa_input,
                                           state_tuple)
        else:
            logit = model(candidate_location, peaks_location, peaks_intensity, precursormass)
            new_state_tuple = None
        logit = torch.squeeze(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        return log_prob, new_state_tuple

    def initial_hidden_state(self, spectrum_representation):
        """

        :param: spectrum_representation, [batch, embedding_size]
        :return:
            [num_lstm_layers, batch_size, lstm_units], [num_lstm_layers, batch_size, lstm_units],
        """
        with torch.no_grad():
            h_0, c_0 = self.init_net(spectrum_representation)
            return h_0.to(device), c_0.to(device)
# model = GraphConvolution(input_dim=3, output_dim=4)
# print(model)
# adj = torch.randn(3, 3, 3)
# input_feat = torch.randn(3, 8, 3, 3)
# print(adj.shape, input_feat.shape)
# output = model(adj, input_feat)
# print(output.size())
#
# model2 = denovo_GCN(input_dim=313, output_dim=26, massAA_np=mass_ID_np).to(device)
# print(model2.parameters())
# location_index = torch.randn(16, 10, 26, 12).to(device)
# peaks_location = torch.randn(16, 300).to(device)
# peaks_intensity = torch.randn(16, 300).to(device)
# start_time = time.time()
# for i in range(1000):
#     output = model2(location_index, peaks_location, peaks_intensity)
#     print(output.size())
# print(time.time() - start_time)
