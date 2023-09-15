#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：OpenABC
@File    ：QoR_predictor.py
@IDE     ：PyCharm
@Author  ：Chenyang Lv
@Date    ：2023/4/30 22:01
'''
import copy

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree
from transformer import *
import os
import os.path as osp

from utils import parseAIGBenchAndCreateNetworkXGraph, pygDataFromNetworkx
import random

allowable_synthesis_features = {
    'synth_type': [0, 1, 2, 3, 4, 5, 6]
}




def get_synth_feature_dims():
    return list(map(len, [
        allowable_synthesis_features['synth_type']
    ]))


full_synthesis_feature_dims = get_synth_feature_dims()

allowable_features = {
    'node_type': [0, 1, 2],
    'num_inverted_predecessors': [0, 1, 2]
}


def get_node_feature_dims():
    return list(map(len, [
        allowable_features['node_type']
    ]))


full_node_feature_dims = get_node_feature_dims()


class NodeEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(NodeEncoder, self).__init__()
        self.node_type_embedding = torch.nn.Embedding(full_node_feature_dims[0], emb_dim)
        torch.nn.init.xavier_uniform_(self.node_type_embedding.weight.data)

    def forward(self, x):
        # First feature is node type, second feature is inverted predecessor
        x_embedding = self.node_type_embedding(x[:, 0])
        # for i in range(1, x.shape[1]):
        # print(x_embedding,x_embedding.shape)
        x_embedding = torch.cat((x_embedding, x[:, 1].reshape(-1, 1)), dim=1)
        return x_embedding


class GCNConv(MessagePassing):
    def __init__(self, in_emb_dim, out_emb_dim):
        super(GCNConv, self).__init__(aggr='add')
        self.linear = torch.nn.Linear(in_emb_dim, out_emb_dim)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self.linear(x)
        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, node_encoder, num_layer, input_dim, emb_dim, gnn_type='gcn'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''
        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.node_emb_size = input_dim
        self.node_encoder = node_encoder

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.convs.append(GCNConv(input_dim, emb_dim))
        self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(1, num_layer):
            self.convs.append(GCNConv(emb_dim, emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):

        # gate_type, node_type, edge_index = batched_data.gate_type, batched_data.node_type, batched_data.edge_index
        edge_index = batched_data.edge_index

        x = torch.cat([batched_data.node_type.reshape(-1, 1), batched_data.num_inverted_predecessors.reshape(-1, 1)],
                      dim=1)
        h = self.node_encoder(x)

        for layer in range(self.num_layer):

            h = self.convs[layer](h, edge_index)
            h = self.batch_norms[layer](h)

            if layer != self.num_layer - 1:
                h = F.relu(h)

        return h


class GNN(torch.nn.Module):

    def __init__(self, node_encoder, input_dim, num_layer=2, emb_dim=128, gnn_type='gcn', graph_pooling="mean"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.graph_pooling = graph_pooling

        self.gnn_node = GNN_node(node_encoder, num_layer, input_dim, emb_dim, gnn_type=gnn_type)
        self.pool1 = global_mean_pool
        self.pool2 = global_max_pool

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph1 = self.pool1(h_node, batched_data.batch)
        h_graph2 = self.pool2(h_node, batched_data.batch)
        return torch.cat([h_graph1, h_graph2], dim=1)


class SynthFlowEncoder(torch.nn.Module):  # the most part to be improved
    def __init__(self, emb_dim):
        super(SynthFlowEncoder, self).__init__()
        self.synth_emb = torch.nn.Embedding(full_synthesis_feature_dims[0], emb_dim)
        torch.nn.init.xavier_uniform_(self.synth_emb.weight.data)

    def forward(self, x):
        # print("before synth_flow encoding:", x.shape)
        # print(x.shape)
        x_embedding = self.synth_emb(x[:, 0])
        for i in range(1, x.shape[1]):
            x_embedding = torch.cat((x_embedding, self.synth_emb(x[:, i])), dim=1)
        # print("after synth_flow encoding:", x_embedding.shape)

        return x_embedding


class SynthConv(torch.nn.Module):
    def __init__(self, inp_channel=1, out_channel=3, ksize=6, stride_len=1):
        super(SynthConv, self).__init__()
        self.conv1d = torch.nn.Conv1d(inp_channel, out_channel, kernel_size=(ksize,), stride=(stride_len,))

    def forward(self, x):
        x = x.reshape(-1, 1, x.size(1))  # Convert [4,60] to [4,1,60]
        x = self.conv1d(x)
        return x.reshape(x.size(0), -1)  # Convert [4,3,55] to [4,165]


class SynthNet_GCN_transformer(torch.nn.Module):  # main model

    def __init__(self, node_encoder, synth_encoder, n_classes, synth_input_dim, node_input_dim, gnn_embed_dim=256,
                 num_fc_layer=3, hidden_dim=128):
        super(SynthNet_GCN_transformer, self).__init__()
        self.num_layers = num_fc_layer
        self.hidden_dim = hidden_dim
        self.node_encoder = node_encoder
        self.synth_encoder = synth_encoder
        self.node_enc_outdim = node_input_dim
        self.synth_enc_outdim = synth_input_dim
        self.gnn_emb_dim = gnn_embed_dim
        self.n_classes = n_classes

        # Synthesis Convolution parameters
        # output_dim = {(input_dim - kernel_size + 2* padding) / stride} + 1
        self.synconv_in_channel = 1
        self.synconv_out_channel = 1
        self.synconv_stride_len = 3

        # Synth Conv1 output
        self.synconv1_ks = 6
        self.synconv1_out_dim_flatten = 1 + (self.synth_enc_outdim - self.synconv1_ks) / self.synconv_stride_len

        # Synth Conv2 output
        self.synconv2_ks = 9
        self.synconv2_out_dim_flatten = 1 + (self.synth_enc_outdim - self.synconv2_ks) / self.synconv_stride_len

        # Synth Conv3 output
        self.synconv3_ks = 12
        self.synconv3_out_dim_flatten = 1 + (self.synth_enc_outdim - self.synconv3_ks) / self.synconv_stride_len

        # Multiplier by 2 since each gate and node type has same encoding out dimension
        # self.gnn = GNN(self.node_encoder,self.node_enc_outdim*2)
        # Node encoding has dimension 3 and number of incoming inverted edges has dimension 1
        self.gnn = GNN(self.node_encoder, self.node_enc_outdim + 1)
        self.synth_conv1 = SynthConv(self.synconv_in_channel, self.synconv_out_channel, ksize=self.synconv1_ks,
                                     stride_len=self.synconv_stride_len)
        self.synth_conv2 = SynthConv(self.synconv_in_channel, self.synconv_out_channel, ksize=self.synconv2_ks,
                                     stride_len=self.synconv_stride_len)
        self.synth_conv3 = SynthConv(self.synconv_in_channel, self.synconv_out_channel, ksize=self.synconv3_ks,
                                     stride_len=self.synconv_stride_len)

        # Multiplier by 2 since each gate and node type has same encoding out dimension
        # self.gnn = GNN(self.node_encoder,self.node_enc_outdim*2)
        # Node encoding has dimension 3 and number of incoming inverted edges has dimension 1

        self.fcs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # GNN + (synthesis flow encoding + synthesis convolution)
        self.in_dim_to_fcs = int(
            self.gnn_emb_dim + 80)  # 256 + 80 = 336  hidden_dim = 128
        self.fcs.append(torch.nn.Linear(self.in_dim_to_fcs, self.hidden_dim))

        for layer in range(1, self.num_layers - 1):
            self.fcs.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))

        self.fcs.append(torch.nn.Linear(self.hidden_dim, self.n_classes))
        ###################
        self.d_model = 4
        self.dropout = 0.1
        self.d_ff = 32
        self.heads = 2
        self.src_tokens = 20
        self.N = 3
        self.attn = MultiHeadedAttention(self.heads, self.d_model)
        self.ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        self.position = PositionalEncoding(self.d_model, self.dropout)
        self.src_embed = nn.Sequential(Embeddings(self.d_model, self.src_tokens), self.position)
        self.encoder = Encoder(EncoderLayer(self.d_model, self.attn, self.ff, self.dropout), self.N)

        self.mask = subsequent_mask(self.src_tokens)
        self.mask = self.mask.to(device='cpu')

    # self.encoder(self.src_embed(src), src_mask)
    def forward(self, batch_data, synVec):
        graphEmbed = self.gnn(batch_data)

        #synthFlow = batch_data.synVec
        synthFlow = synVec
        #print('synthFlow:', synthFlow.shape)  # h_syn: torch.Size([1, 20, 50])
        #print(synthFlow)

        # Synthesis flow length = 20
        # h_syn = self.synth_encoder(synthFlow.reshape(-1, 20))
        embeded_syn = self.src_embed(synthFlow.reshape(-1, 20))
        #print('embeded_syn:', embeded_syn.shape)  # h_syn: torch.Size([1, 20, 50])
        # print(embeded_syn)
        transformer_encode_outcome = self.encoder(embeded_syn, self.mask)
        # print('transformer_encode_outcome:', transformer_encode_outcome.shape)
        # print(transformer_encode_outcome)
        # synconv1_out = self.synth_conv1(h_syn)
        # print('synconv1_out:', synconv1_out.shape)
        # synconv2_out = self.synth_conv2(h_syn)
        # print('synconv1_out:', synconv2_out.shape)
        # synconv3_out = self.synth_conv3(h_syn)
        # print('synconv1_out:', synconv3_out.shape)

        syn_embed = transformer_encode_outcome.flatten(start_dim=1, end_dim=2)
        #print('syn_embed.shape', syn_embed.shape)

        # transformer_encode_outcome = torch.tensor([8, 54])
        # print('graphEmbed:', graphEmbed.shape)

        concatenatedInput = torch.cat([graphEmbed, syn_embed], dim=1)
        #print('concatenatedInput', concatenatedInput.shape)
        # concatenatedInput = torch.cat([graphEmbed, synconv1_out, synconv2_out, synconv3_out], dim=1)

        # print('concatenatedInput:', concatenatedInput.shape)
        x = F.relu(self.fcs[0](concatenatedInput))
        for layer in range(1, self.num_layers - 1):
            x = F.relu(self.fcs[layer](x))

        x = self.fcs[-1](x)
        return x
def graph_info_extractor(benchFile):
    INPUT_BENCH = benchFile
    AIG_DAG = parseAIGBenchAndCreateNetworkXGraph(INPUT_BENCH)
    #print(AIG_DAG)
    data = pygDataFromNetworkx(AIG_DAG)
    return data

def check_list(lst):
    temp = copy.deepcopy(lst)
    if len(temp) < 20:
        temp += [random.randint(0,6) for i in range(20 - len(lst))]
    return temp
def check_list2(lst):
    temp = copy.deepcopy(lst)
    while len(temp) < 20:
        temp += temp[:20-len(temp)]
        #temp += [6]
    return temp

import pickle
from torch_geometric.data import DataLoader
def nodes_predict(graph_model, desID, synVec):
    pkl_file_name = 'custom_extracted_graph/' + desID + '-init_graph.plk'
    if os.path.exists(pkl_file_name):  # check if there exist cache
        with open(pkl_file_name, 'rb') as f1:
            InitGraphData = pickle.load(f1)
    else:
        InitGraphData = graph_info_extractor(
            '/home/lcy/PycharmProjects/OpenABC/models/qor/SynthNetV5-GCN+DecisionTransformer/DecisionTransformer/orig/' + desID + '_orig.bench')

        f2 = open(pkl_file_name, 'wb')
        pickle.dump(InitGraphData, f2)

    device = 'cpu'

    synVec_ = copy.deepcopy(synVec)
    synVec_ = check_list2(synVec_)
    print(synVec_)
    new_synVec = torch.tensor(synVec_).to(device)
    loader = DataLoader([InitGraphData], shuffle=True, batch_size=1)
    batch = loader.__iter__().next()
    graph_model.eval()
    predict = graph_model(batch, new_synVec).clone().to('cpu').detach().item()
    return predict

def get_random_list(len):
    return [random.randint(0, 6) for _ in range(len)]

if __name__ == '__main__':
    GRAPH_DIR = "/home/lcy/PycharmProjects/OpenABC/net4variant2/SynthNETV4_set2"
    num_classes = 1
    nodeEmbeddingDim = 3
    synthEncodingDim = 4
    # synthFlowEncodingDim = trainDS[0].synVec.size()[0] * synthEncodingDim  # 60
    synthFlowEncodingDim = 80  # 60
    device = 'cpu'
    graph_node_encoder = NodeEncoder(emb_dim=nodeEmbeddingDim).to(device)
    synthesis_encoder = SynthFlowEncoder(emb_dim=synthEncodingDim).to(device)
    graph_model = SynthNet_GCN_transformer(node_encoder=graph_node_encoder, synth_encoder=synthesis_encoder,
                                           n_classes=num_classes,
                                           synth_input_dim=synthFlowEncodingDim, node_input_dim=nodeEmbeddingDim)
    graph_model.load_state_dict(
        # torch.load(osp.join(DUMP_DIR, 'gcn-epoch-{}-val_loss-{:.3f}.pt'.format(60, 0.063))))
        # torch.load(osp.join(DUMP_DIR, 'gcn-epoch-{}-val_loss-{:.3f}.pt'.format(36, 0.840))))
        # torch.load(osp.join(GRAPH_DIR, 'gcn-epoch-{}-val_loss-{:.3f}.pt'.format(78, 0.515))))
        torch.load(osp.join(GRAPH_DIR, 'gcn-epoch-49-val_loss-0.462.pt'), map_location='cpu'))
    graph_model.eval()
    desID = 'div'
    # self.init_normed_and_nodes = (self.init_and_nodes - self.mean_and) / self.mean_depth
    # self.init_normed_depth = (self.init_depth - self.meanVarTargetDict[self.desID][2]) / self.meanVarTargetDict[self.desID][3]
    synVec = [1]  # imprv 52.99
    print(nodes_predict(graph_model, desID, synVec))
    synVec = [1, 5, 6, 1, 0]  # imprv 52.99
    print(nodes_predict(graph_model, desID, synVec))
    synVec = [1, 5, 6, 1, 0, 5, 1, 1, 6, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3]  # imprv 52.99
    print(nodes_predict(graph_model, desID, synVec))
    synVec = [6, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 6, 6, 6, 6, 6, 4, 4, 4, 4]  # imprv 43.21
    print(nodes_predict(graph_model, desID, synVec))
    synVec = [1, 5, 4, 6, 4, 4, 3, 1, 6, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4]  # imprv 43.56
    print(nodes_predict(graph_model, desID, synVec))
    synVec = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # imprv ???
    print(nodes_predict(graph_model, desID, synVec))
    synVec = [6, 2, 0, 6,2,3,6,1,3,6,6,2,0,6,2,3,6,1,3,6]  # resyn2 imprv
    print(nodes_predict(graph_model, desID, synVec))