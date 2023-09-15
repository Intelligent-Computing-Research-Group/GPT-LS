#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from subprocess import check_output
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, degree
import random
from transformer import *
from utils import *
from QoR_predictor import nodes_predict
from torch_geometric.data import DataLoader

allowable_synthesis_features = {
    'synth_type': [0, 1, 2, 3, 4, 5, 6]
}

def make_abc_commands(actions, input_file_name, input_design_name, extended_mode, output_file_name=None): # do not use scl
    action_ = actions[-1:]
    abc_command = ""
    abc_command += "read " + input_file_name + "; "
    if input_file_name.endswith('aig'):
        pass
    else:
        abc_command += "strash; "  # convert the input file into aig type
    for i in action_:
        abc_command += takeAction2(i) + "; "

    if not os.path.exists(output_file_name):
        abc_command += 'write ' + output_file_name
        abc_command += '; write_bench -l ' + output_file_name[:-3] + 'bench'
    abc_command += '; print_stats'
    return abc_command

def get_metrics2(stats):
    """
    parse delay and area from the stats command of ABC
    """
    line = stats.decode("utf-8").split('\n')[-2].split(':')[-1].strip()
    #print('line2:', line)
    ob = re.search(r'and *= *[0-9]+.?[0-9]*', line)
    #print('ob:', ob)
    NumAnd = float(ob.group().split('=')[1].strip())
    ob = re.search(r'lev *= *[0-9]+.?[0-9]*', line)
    lev = float(ob.group().split('=')[1].strip())
    return lev, NumAnd

def make_abc_baseline_commands(actions, input_file_name, input_design_name, output_file_name=None): # do not use scl
    # abc_command = "read stdcells.lib; "  # this file is standard cell library
    #abc_command = "read asap7.lib; "  # this file is standard cell library
    abc_command = ""
    abc_command += "read " + input_file_name + "; "
    abc_command += "strash; "  # convert the input file into aig type
    for i in actions:
        abc_command += takeAction3(i) + "; "
        #print(takeAction2(i))
    # if output_file_name is not None:
    #     abc_command += " write " + output_file_name + "; "
    #     #print('is not none')
    # else:
    #     abc_command += ""
        #print('is none')

    #abc_command += 'write_bench -l ' + input_design_name + '.bench; '
    abc_command += 'print_stats'
    return abc_command

def takeAction2(actionIdx):
    if actionIdx == 0:
        return 'refactor' #rf
    elif actionIdx == 1:
        return 'refactor -z' #rf -z
    elif actionIdx == 2:
        return 'rewrite'  # rw
    elif actionIdx == 3:
        return 'rewrite -z'  # rw -z
    elif actionIdx == 4:
        return 'resub'  # rs
    elif actionIdx == 5:
        return 'resub -z' #rs-z
    elif actionIdx == 6:
        return 'balance' #b
    else:
        assert (False)

def takeAction3(actionIdx):
    if actionIdx == 0:
        return 'resyn'  # "b; rw; rwz; b; rwz; b"
    elif actionIdx == 1:
        return 'resyn2'  # "b; rw; rf; b; rw; rwz; b; rfz; rwz; b" "6,2,0,6,2,3,6,1,3,6"
    elif actionIdx == 2:
        return 'resyn2a'  # "b; rw; b; rw; rwz; b; rwz; b"
    elif actionIdx == 3:
        return 'resyn3'  # "b; rs; rs -K 6; b; rsz; rsz -K 6; b; rsz -K 5; b"
    elif actionIdx == 4:
        return 'compress'  # "b -l; rw -l; rwz -l; b -l; rwz -l; b -l"
    elif actionIdx == 5:
        return 'compress2'  # "b -l; rw -l; rf -l; b -l; rw -l; rwz -l; b -l; rfz -l; rwz -l; b -l"
    else:
        assert (False)
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


    def forward(self, batch_data, synVec):
        graphEmbed = self.gnn(batch_data)
        synthFlow = synVec
        embeded_syn = self.src_embed(synthFlow.reshape(-1, 20))
        transformer_encode_outcome = self.encoder(embeded_syn, self.mask)
        syn_embed = transformer_encode_outcome.flatten(start_dim=1, end_dim=2)
        concatenatedInput = torch.cat([graphEmbed, syn_embed], dim=1)

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

import pickle


numSynthesized = 1

import argparse,os
import sys
INPUT_BENCH = None
GML_DUMP_LOC = None



class abc_env():  # meanVarTargetDict is the result of 20_random_commands
    def __init__(self, desID, desFile, use_graph_info, gnn_model, extended_mode=False):
        self.desID = desID
        print("design:", self.desID)

        self.des_file = desFile
        self.default_des_file = desFile
        self.actions = []
        #self.action_str = ''
        temp_dir = "playground/" + str(self.desID)
        temp_dir_extend = "playground_extend/" + str(self.desID)
        temp_dir_state_predict = "playground_state_predict/" + str(self.desID)
        temp_dir_ground_truth = "playground_ground_truth/" + str(self.desID)
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        if extended_mode:
            if not os.path.exists(temp_dir_extend):
                os.mkdir(temp_dir_extend)
        if not os.path.exists(temp_dir_state_predict):
            os.mkdir(temp_dir_state_predict)
        if not os.path.exists(temp_dir_ground_truth):
            os.mkdir(temp_dir_ground_truth)
        abc_command = make_abc_commands(self.actions, self.des_file, self.desID, output_file_name="playground/" + str(self.desID) + "/" + str(self.desID) + "-.aig", extended_mode=False)
        # print('abc_command:', abc_command)
        proc = check_output(['yosys-abc', '-c', abc_command])
        #print('Init_proc:', proc)
        Lev, NumAnd = get_metrics2(proc)
        self.init_and_nodes = NumAnd
        self.init_depth = Lev
        self.and_nodes = 0  # current
        self.depth = 0  # current
        self.normed_and_nodes = 0
        self.normed_depth = 0
        self.gnn_model = gnn_model
        self.extended_mode = extended_mode
        self.reward = 1
        #self.mean_and = meanVarTargetDict[desID][0]
        self.mean_and = NumAnd
        #print('mean_and:', self.mean_and)
        #self.mean_depth = meanVarTargetDict[desID][2]
        self.mean_depth = Lev
        #print('mean_depth:', self.mean_depth)
        #self.meanVarTargetDict = meanVarTargetDict
        self.runtime_for_graph_extract = 0
        self.runtime_for_synthesis = 0
        self.use_graphinfo = use_graph_info
        time1 = time.time()
        self.InitGraphData = self.graph_info_extractor('./orig/' + self.desID + '_orig.bench')

        self.CurGraphData = self.InitGraphData
        #print('Init GraphInfo', self.InitGraphData)

        # self.init_normed_and_nodes = (self.init_and_nodes - self.mean_and) / self.mean_depth
        # self.init_normed_depth = (self.init_depth - self.meanVarTargetDict[self.desID][2]) / self.meanVarTargetDict[self.desID][3]
        actions = []

        #self.CurGraphData = self.graph_info_extractor('temp.bench')
        #print(self.CurGraphData)
        loader = DataLoader([self.CurGraphData], shuffle=True, batch_size=1)
        batch = loader.__iter__().next()
        self.init_graph_tensor = self.gnn_model(batch).clone().detach()
        time2 = time.time()
        runtime_graph_extract = time2 - time1
        self.runtime_for_graph_extract += runtime_graph_extract
        print('runtime of init graph extract:', runtime_graph_extract)
        self.default_graph_tensor = self.init_graph_tensor.clone().detach()
        #print('init_graph_tensor', self.init_graph_tensor.shape)

        pkl_file_name = "training_set_mean_state.pkl"
        if os.path.exists(pkl_file_name):  # check if there exist cache
        #if False:  # check if there exist cache
            with open(pkl_file_name, 'rb+') as f1:
                self.mean_state, self.std_state = pickle.load(f1)
        else:
            train_list = ['spi', 'i2c', 'ss_pcm', 'usb_phy', 'sasc', 'wb_dma', 'simple_spi', 'pci', 'ac97_ctrl',
                          'mem_ctrl', 'des3_area', 'sha256', 'fir', 'iir', 'tv80', 'dynamic_node']
            state_list = []
            for des in train_list:
                state_list.append(self.get_graph_state(des))
            self.mean_state = torch.mean(torch.stack(state_list), dim=0)
            self.std_state = torch.std(torch.stack(state_list), dim=0)
            with open(pkl_file_name, 'wb+') as f3:
                pickle.dump([self.mean_state, self.std_state], f3)


        self.graph_state_mean = torch.mean(self.init_graph_tensor)
        # print('init_graph_tensor_mean', self.graph_state_mean)
        self.graph_state_std = torch.std(self.init_graph_tensor)
        # print('init_graph_tensor_std', self.graph_state_std)
        time3 = time.time()
        baseline_actions = [1, 1]  # resyn2 for 2 times
        abc_command = make_abc_baseline_commands(baseline_actions, self.des_file, self.desID)  # abc.rc must in current dic
        # print('abc_command:', abc_command)
        proc = check_output(['yosys-abc', '-c', abc_command])
        #print('Baseline_proc:', proc)
        Lev_baseline, NumAnd_baseline = get_metrics2(proc)
        self.rtg_baseline = 1-(NumAnd_baseline / self.init_and_nodes)
        print('Baseline_qor:', self.rtg_baseline)
        time4 = time.time()
        time_resyn2 = time4 - time3
        print('runtime of resyn2*2:', time_resyn2)
        GRAPH_DIR = "./net4variant2/SynthNETV4_set2"
        num_classes = 1
        nodeEmbeddingDim = 3
        synthEncodingDim = 4
        # synthFlowEncodingDim = trainDS[0].synVec.size()[0] * synthEncodingDim  # 60
        synthFlowEncodingDim = 80  # 60
        device = 'cpu'
        graph_node_encoder = NodeEncoder(emb_dim=nodeEmbeddingDim).to(device)
        synthesis_encoder = SynthFlowEncoder(emb_dim=synthEncodingDim).to(device)
        self.qor_predict_model = SynthNet_GCN_transformer(node_encoder=graph_node_encoder, synth_encoder=synthesis_encoder,
                                               n_classes=num_classes,
                                               synth_input_dim=synthFlowEncodingDim, node_input_dim=nodeEmbeddingDim)
        self.qor_predict_model.load_state_dict(
            # torch.load(osp.join(DUMP_DIR, 'gcn-epoch-{}-val_loss-{:.3f}.pt'.format(60, 0.063))))
            # torch.load(osp.join(DUMP_DIR, 'gcn-epoch-{}-val_loss-{:.3f}.pt'.format(36, 0.840))))
            # torch.load(osp.join(GRAPH_DIR, 'gcn-epoch-{}-val_loss-{:.3f}.pt'.format(78, 0.515))))
            torch.load(os.path.join(GRAPH_DIR, 'gcn-epoch-49-val_loss-0.462.pt'), map_location='cpu'))
        self.qor_predict_model.eval()
        self.resyn2qor_pred = -nodes_predict(self.qor_predict_model, self.desID, [6,2,0,6,2,3,6,1,3,6,6,2,0,6,2,3,6,1,3,6])
        print('resyn2qor_pred:', self.resyn2qor_pred)
        self.iterative_mode = False

    def get_graph_state(self, desID):
        InitGraphData = self.graph_info_extractor(
            './orig/' + desID + '_orig.bench')

        CurGraphData = InitGraphData
        print('Init GraphInfo', InitGraphData)

        # self.init_normed_and_nodes = (self.init_and_nodes - self.mean_and) / self.mean_depth
        # self.init_normed_depth = (self.init_depth - self.meanVarTargetDict[self.desID][2]) / self.meanVarTargetDict[self.desID][3]

        # self.CurGraphData = self.graph_info_extractor('temp.bench')
        # print(self.CurGraphData)
        loader = DataLoader([CurGraphData], shuffle=True, batch_size=1)
        batch = loader.__iter__().next()
        graph_tensor = self.gnn_model(batch).clone().detach()
        return graph_tensor
    def reset(self):
        #self.and_nodes = self.init_and_nodes
        #self.depth = self.init_depth
        #self.actions = []
        #abc_command = make_abc_commands(self.actions, self.des_file, self.desID)
        # print('abc_command:', abc_command)
        #proc = check_output(['yosys-abc', '-c', abc_command])
        #print('Reset_proc:', proc)
        #Lev, NumAnd = get_metrics2(proc)
        # abc_command = make_abc_commands(self.actions, self.des_file)
        # #print('abc_command:', abc_command)
        # proc = check_output(['yosys-abc', '-c', abc_command])
        # #print('proc:', proc)
        # Lev, NumAnd = get_metrics2(proc)
        # self.and_nodes = NumAnd
        # self.depth = Lev

        #done = False
        #return torch.tensor([self.init_normed_and_nodes, self.init_normed_depth], dtype=torch.float32)
        #print('init_graph', self.init_graph_tensor)
        return self.init_graph_tensor.clone().detach()
    def default_setting(self):
        self.des_file = self.default_des_file
        self.init_graph_tensor = self.default_graph_tensor
    def change_init_env(self, input_file):
        self.des_file = input_file
        InitGraphData = self.graph_info_extractor(
            input_file)

        CurGraphData = InitGraphData
        #print('Init GraphInfo',InitGraphData)

        # self.init_normed_and_nodes = (self.init_and_nodes - self.mean_and) / self.mean_depth
        # self.init_normed_depth = (self.init_depth - self.meanVarTargetDict[self.desID][2]) / self.meanVarTargetDict[self.desID][3]
        actions = []

        # self.CurGraphData = self.graph_info_extractor('temp.bench')
        # print(self.CurGraphData)
        loader = DataLoader([CurGraphData], shuffle=True, batch_size=1)
        batch = loader.__iter__().next()
        self.init_graph_tensor = self.gnn_model(batch).clone().detach()
    def step(self, actions, skip_graph_extraction):
        action_str = ''

        if len(actions)==0:
            print('len(actions)==0')
            input_file_name = "playground/" + str(self.desID) + "/" + str(self.desID) + "-.aig"
        else:
            for act in actions[:-1]:
                action_str += str(act)
            input_file_name = "playground/" + str(self.desID) + "/" + str(self.desID) + "-" + action_str + ".aig"
        action_str1 = ''
        for act in actions:
            action_str1 += str(act)
        output_file_name = "playground/" + str(self.desID) + "/" + str(self.desID) + "-" + action_str1 + ".aig"
        #self.actions.append(actions[-1])
        #abc_command = make_abc_commands([self.actions[-1]], self.desID + '.bench', self.desID)

        abc_command = make_abc_commands(actions, input_file_name, self.desID, output_file_name=output_file_name, extended_mode=self.extended_mode)
        #print('abc_command:', abc_command)
        start_syn = time.time()
        proc = check_output(['yosys-abc', '-c', abc_command])  # if pre_actions exist, abc_command changes
        #print('proc:', proc)
        Lev, NumAnd = get_metrics2(proc)
        end_syn = time.time()
        sum_reward = (self.init_and_nodes - NumAnd) / self.init_and_nodes
        #self.sum_reward = sum_reward
        during_syn = end_syn - start_syn
        self.runtime_for_synthesis += during_syn
        #print(self.runtime_for_synthesis)
        #print(NumAnd, end=' ')
        #sys.stdout.flush()
        last_and_nodes = self.and_nodes
        last_depth = self.depth
        self.and_nodes = NumAnd
        self.depth = Lev
        # self.normed_and_nodes = (NumAnd - self.meanVarTargetDict[self.desID][0])/self.meanVarTargetDict[self.desID][1]
        # self.normed_depth = (Lev - self.meanVarTargetDict[self.desID][2])/self.meanVarTargetDict[self.desID][3]
        reward = (last_and_nodes - NumAnd)/self.init_and_nodes
        self.reward = reward

        #print('sum_reward:', sum_reward)
        done = False
        #print('cur action length:', len(self.actions))
        if len(actions) >= 20:
            done = True

        if skip_graph_extraction:
            return self.reset(), reward, done, sum_reward
        flag = True
        while flag:
            try:
                start = time.time()
                self.CurGraphData = self.graph_info_extractor(output_file_name[:-3]+'bench')
                loader = DataLoader([self.CurGraphData], shuffle=True, batch_size=1)
                batch = loader.__iter__().next()
                graph_state = self.gnn_model(batch)
                flag = False
                end = time.time()
                during = end - start
                self.runtime_for_graph_extract += during
                return graph_state, self.reward, done, sum_reward
            except:
                print('reload graph_info')  #
                time.sleep(random.uniform(1, 3))
        #print(self.CurGraphData)
        #print('graph_state:', graph_state.shape)

    def graph_info_extractor(self, benchFile):
        #print(benchFile)
        INPUT_BENCH = benchFile
        AIG_DAG = parseAIGBenchAndCreateNetworkXGraph(INPUT_BENCH)
        data_ = pygDataFromNetworkx(AIG_DAG)
        flag = False
        #print(AIG_DAG)
        return data_

