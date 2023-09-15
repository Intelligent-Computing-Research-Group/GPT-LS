#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：GPT-LS
@File    ：decision_transformer.py
@IDE     ：PyCharm
@Author  ：Chenyang Lv
@Date    ：2023/3/13 下午9:37
'''

import numpy as np
import torch
import torch.nn as nn
import transformers
from trajectory_gpt2 import GPT2Model
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

allowable_features = {
    'node_type': [0, 1, 2],
    'num_inverted_predecessors': [0, 1, 2]
}
device = 'cpu'

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
        #print("before node encoding:", x.shape)
        # print(x.shape)
        x_embedding = self.node_type_embedding(x[:, 0])
        x_embedding = torch.cat((x_embedding, x[:, 1].reshape(-1, 1)), dim=1)
        #print("after node encoding:", x_embedding.shape)
        # print(x_embedding.shape)
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

    def forward(self, batched_data):  # process graph info

        # gate_type, node_type, edge_index = batched_data.gate_type, batched_data.node_type, batched_data.edge_index
        edge_index = batched_data.edge_indexs

        x = torch.cat([batched_data.node_types.reshape(-1, 1), batched_data.num_inverted_predecessorses.reshape(-1, 1)],
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
        #print('batched_data.batch', batched_data.batch)
        h_graph1 = self.pool1(h_node, batched_data.batch)
        h_graph2 = self.pool2(h_node, batched_data.batch)
        return torch.cat([h_graph1, h_graph2], dim=1)



class TrajectoryModel(nn.Module):

    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    # def forward(self, states, actions, rewards, masks=None, attention_mask=None):
    #     # "masked" tokens or unspecified inputs can be passed in as None
    #     return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(actions[-1])

class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,

            max_length=20,
            max_ep_len=20,
            action_tanh=False,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_ctx=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)

        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)

        nodeEmbeddingDim = 3
        self.node_encoder = NodeEncoder(emb_dim=nodeEmbeddingDim)
        node_input_dim = 3
        self.node_enc_outdim = node_input_dim
        self.gnn = GNN(self.node_encoder, self.node_enc_outdim + 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, batch_size, attention_mask=None):
        #states = torch.reshape(states, ((1, 20, 2))).to('cuda')
        #print('states.shape:', states.shape)

        #Graph_state = self.gnn(Graph)
        #print('Graph_state:', Graph_state.shape)

        #concatenatedInput = torch.cat([Graph_state, states], dim=-1)
        #print('concatenatedInput:', concatenatedInput.shape)


        #print('actions.shape', actions.shape)

        states = torch.reshape(states, (batch_size, -1, self.state_dim))
        # print('states.shape:', states.shape)# 1, 20, 256
        actions = torch.reshape(actions, (batch_size, -1, self.act_dim))
        #print('actions:', actions)
        #print('actions.shape', actions.shape)# 1, 20, 7
        timesteps = torch.reshape(timesteps, (batch_size, -1))
        #print('timesteps:', timesteps.shape)
        #input()
        returns_to_go = torch.reshape(returns_to_go, (batch_size, -1, 1))
        #print('returns_to_go:', returns_to_go.shape)# 1, 20, 1
        #print(returns_to_go)
        #input()
        #timesteps = torch.reshape(timesteps, ((20, 1))).to('cuda')
        #print(states.shape[0])
        #print(states.shape[1])

        seq_length = states.shape[1]
        #print('seq_length', seq_length)
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(device)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states.to(device))
        action_embeddings = self.embed_action(actions.to(device))
        returns_embeddings = self.embed_return(returns_to_go.to(device))
        time_embeddings = self.embed_timestep(timesteps.to(device))
        # print('returns_embeddings:', returns_embeddings.shape) # 1, 20, 128
        # print('state_embeddings:', state_embeddings.shape)# 1, 20, 128
        # print('action_embeddings:', action_embeddings.shape)# 1, 20, 128
        # print('time_embeddings:', time_embeddings.shape)# 1, 20, 128
        # input()
        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings

        #print('state_embeddings:', state_embeddings.shape)

        #print('action_embeddings:', action_embeddings.shape)
        action_embeddings = action_embeddings + time_embeddings

        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        #print('stacked_inputs:', stacked_inputs.shape) # 1, 60, 128
        stacked_inputs = self.embed_ln(stacked_inputs)
        #print('stacked_inputs:', stacked_inputs.shape) # 1, 60, 128
        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)
        #print('stacked_attention_mask:', stacked_attention_mask.shape)#torch.Size([1, 60])
        #input()
        #print('attention_mask:', attention_mask)
        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        #print('x.shape:', x.shape) #torch.Size([1, 60, 128])
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        #print('x.shape:', x.shape) # torch.Size([1, 3, 20, 128])
        # get predictions
        return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state
        #input()
        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])  # 0000000000001 at first
            #attention_mask = torch.ones(self.max_length)  # 0000000000001 at first
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None
        #print('states:', states.shape)
        #print('states:', states)
        #print('actions:', actions)
        # print('returns_to_go:', returns_to_go.shape)
        state_preds, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, batch_size=1, attention_mask=attention_mask, **kwargs)
        #print('action_preds:',action_preds)
        #print('return_preds:',return_preds)
        #input()
        return action_preds[0, -1], state_preds[0, -1], return_preds[0, -1]
