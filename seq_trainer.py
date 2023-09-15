import numpy as np
import torch
import torch.nn.functional as F
from trainer import Trainer
device = 'cpu'

def subsequent_mask(size_):
    "Mask out subsequent positions."
    attn_shape = (1, size_, size_)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')  #
    return torch.from_numpy(subsequent_mask) == 0
class SequenceTrainer(Trainer):

    def train_step(self, batch):
        #batch = batch.to(device)
        action_labal = batch.synVec
        actions = torch.zeros(20*self.batch_size, self.act_dim).to(device).scatter(1, action_labal.unsqueeze(-1), 1)
        #print("a.shape:", actions.shape)
        #d = batch.dones
        #states = batch.states.unsqueeze(dim=0)
        #action_labal = action_labal.reshape(actions, (batch_size, -1, self.act_dim))
        Graph = batch.graph_state.clone().detach()
        #Graph_next_true_state = batch.graph_state_pred
        # print('Graph shape:', Graph.shape)
        # print('Graph_next_true_stateshape:', Graph_next_true_state.shape)


        timesteps = batch.timestep
        rtg = batch.rtg_list.unsqueeze(dim=1)
        rewards = batch.rtg_list.unsqueeze(dim=1) #useless
        #print('states:', states.shape)
        # states, actions, rewards, dones, rtg, timesteps, attention_mask, Graph = batch
        action_target = torch.clone(actions).to(device)
        # print(states)
        #print('actions', actions.shape)
        # print(rewards)
        # print(dones)
        # print(rtg)
        #print(rtg[:,:-1])
        #print(returns)
        # batch_size, seq_length = states.shape[0], states.shape[1]
        # print('batch_size', batch_size)
        # print('seq_length', seq_length)
        #print('action shape:', actions.shape)
        #print('action shape:', actions.shape)
        #print('rtg shape:', rtg.shape)
        #print('timesteps', timesteps.shape[-1])
        attention_mask = torch.ones((self.batch_size, timesteps.shape[-1]), dtype=torch.long).to(device)
        state_preds, action_preds, reward_preds = self.model.forward(
            Graph, actions, rewards, rtg, timesteps, self.batch_size, attention_mask=attention_mask,
        )
        # print('state_preds', state_preds.shape)
        # print('action_preds', action_preds.shape)
        # print('reward_preds', reward_preds.shape)

        act_dim = action_preds.shape[2]
        #print('act_dim:', act_dim)
        #print(action_preds)
        #print(action_preds.shape)
        action_preds = torch.squeeze(action_preds)
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        # print('action_preds:', action_preds)
        # print('action_target:', action_target)
        # input()
        # loss = self.loss_fn(
        #     None, action_preds, None,
        #     None, action_target, None,
        # )
        state_preds = torch.squeeze(state_preds)
        state_preds = state_preds.reshape(-1, self.state_dim)[attention_mask.reshape(-1) > 0]

        labels = action_labal.squeeze(dim=-1)
        # print(action_preds.shape)
        # print(labels.shape)
        loss = F.cross_entropy(action_preds, labels)
        #MSE_loss = torch.nn.MSELoss()

        # print(state_preds.shape)
        # print(Graph_next_true_state.shape)
        #loss_graph_state = MSE_loss(state_preds, Graph_next_true_state)
        #loss_reward = MSE_loss(state_preds, Graph_next_true_state)
        print('loss_action_pred:', loss)
        #print('loss_graph_state:', loss_graph_state)
        #concentrated_loss = loss_graph_state + loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()
