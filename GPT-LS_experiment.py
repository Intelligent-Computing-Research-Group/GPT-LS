import numpy as np
import torch
import os.path as osp
import argparse
import pickle
import random

from torchvision import transforms
from evaluate_episodes import *
from decision_transformer import DecisionTransformer
#from mlp_bc import MLPBCModel
#from act_trainer import ActTrainer
from seq_trainer import SequenceTrainer
from torch_geometric.data import Dataset, DataLoader
from zipfile import ZipFile
from tqdm import tqdm
import pandas as pd
from utils import *
from env import abc_env
import os
#from graph_extractor import *
from graph_extractor_net4 import *

datasetDict = {
    'set1': ["QoR_predict_train_data_set1.csv", "QoR_predict_test_data_set1.csv"],
    'set2': ["train_data_set2.csv", "test_data_set2.csv"],
    'set4': ["QoR_predict_train_fulldata_set1.csv", "QoR_predict_test_fulldata_set1.csv"],
    'set5': ["QoR_predict_train_complete_data_set1.csv", "QoR_predict_test_complete_data_set1.csv"],
    'set3': ["train_data_mixmatch_v1.csv", "test_data_mixmatch_v1.csv"],
    'set6': ["mixed_train_data_set1.csv", "mixed_train_data_set1_extra.csv", 'mixed_train_data_set1_full.csv']
}

DUMP_DIR = "./SynthNETV5_set1-test"
GRAPH_DIR = "./SynthNETV3_set1"
class NetlistGraphDataset(Dataset):
    def __init__(self, root, filePath, transform=None, pre_transform=None):
        self.filePath = osp.join(root, filePath)
        super(NetlistGraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def processed_file_names(self):
        fileDF = pd.read_csv(self.filePath)
        return fileDF['fileName'].tolist()

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):  #
        filePathArchive = osp.join(self.processed_dir, self.processed_file_names[idx])
        filePathName = osp.basename(osp.splitext(filePathArchive)[0])
        with ZipFile(filePathArchive) as myzip:
            with myzip.open(filePathName) as myfile:
                data = torch.load(myfile)
        #print(data)
        #input()
        return data
    # def add_class(self, class_id, class_name):
    #     self.c

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    print("Using device:", device)
    #log_to_wandb = variant.get('log_to_wandb', False)

    dataset = variant['dataset']
    model_type = variant['model_type']
    # using_state_predictor = variant['using_state_predictor']
    # print('using_state_predictor:', using_state_predictor)
    using_state_predictor = False
    print('using_state_predictor:', using_state_predictor)
    beam_width = 2
    final_repeat = 1  # 1 or 25
    group_name = f'{exp_prefix}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    #eval_mode = variant['only_eval_mode']
    eval_mode = True
    print('eval_mode:', eval_mode)
    #eval_mode = False
    # osp.join('/scratch/abc586/OpenABC-dataset/SynthV9_AND',RUN_DIR)

    if not osp.exists(DUMP_DIR):
        os.mkdir(DUMP_DIR)
    # env = gym.make('Hopper-v3')

    max_ep_len = 20
    #env_targets = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    #env_targets = [0.2, 0.1, -0.1]
    #env_targets = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    #env_targets = [50, 10, 5, 2, 1, 0.5, 0.3, 0, -1, -2] #env targets 100, 50, 20,
    #env_targets = [2**i-2 for i in range(0,10,1)] #env targets 100, 50, 20,
    #env_targets = [-0.65, -0.75, -0.9, -1.1, -1.3, -3, -5, -10] #env targets 100, 50, 20,
    #env_targets = [100, 50, 20, 15, 12] #env targets 100, 50, 20,
    #env_targets = [7,6.5, 6,5.5,5,4.5,4,3.5,3] #env targets 100, 50, 20,
    #env_targets = [1, 0.5, 0.3, 0.1, 0] #env targets
    #env_targets = [1000, 500, 200, 100, 50, 20] #env targets
    scale = 1.  # normalization for rewards/returns
    state_dim = 256
    act_dim = 7

    learningProblem = 1
    ROOT_DIR = "/home/lcy/PycharmProjects/OPENABC2_DATASET"
    with open(osp.join("/home/lcy/PycharmProjects/OPENABC2_DATASET", 'synthesisStatistics.pickle'), 'rb') as f:
        targetStats = pickle.load(f)

    trainDS = NetlistGraphDataset(root=ROOT_DIR, filePath=datasetDict["set6"][0])

    trainDS_full = NetlistGraphDataset(root=ROOT_DIR, filePath=datasetDict["set6"][1])

    print(trainDS)
    #print(testDS)
    print(trainDS_full)
    #print(testDS_full)

    num_classes = 1
    nodeEmbeddingDim = 3
    synthEncodingDim = 4
    #synthFlowEncodingDim = trainDS[0].synVec.size()[0] * synthEncodingDim  # 60
    synthFlowEncodingDim = 80  # 60
    graph_node_encoder = NodeEncoder(emb_dim=nodeEmbeddingDim)
    synthesis_encoder = SynthFlowEncoder(emb_dim=synthEncodingDim)
    graph_model = SynthNet_GCN_transformer(node_encoder=graph_node_encoder, synth_encoder=synthesis_encoder, n_classes=num_classes,
                           synth_input_dim=synthFlowEncodingDim, node_input_dim=nodeEmbeddingDim)
    targetLbl = 'nodes'
    meanVarTargetDict = computeMeanAndVarianceOfTargets(targetStats, targetVar=targetLbl)  # normalize the dataset
    GRAPH_DIR = "./net4variant2/SynthNETV4_set2"
    graph_model.load_state_dict(
        # torch.load(osp.join(DUMP_DIR, 'gcn-epoch-{}-val_loss-{:.3f}.pt'.format(60, 0.063))))
        #torch.load(osp.join(DUMP_DIR, 'gcn-epoch-{}-val_loss-{:.3f}.pt'.format(36, 0.840))))
        #torch.load(osp.join(GRAPH_DIR, 'gcn-epoch-{}-val_loss-{:.3f}.pt'.format(78, 0.515))))
        torch.load(osp.join(GRAPH_DIR, 'gcn-epoch-49-val_loss-0.462.pt')))
    graph_model.eval()
    extended_mode = False
    test_des_in_env = variant['design']
    env = abc_env(desID=test_des_in_env, desFile=variant['design_file'], use_graph_info=True,
                  gnn_model=graph_model, extended_mode=extended_mode)

    DATA_SET_FILE = "./"
    RTG_FILE = DATA_SET_FILE + 'rtg_data/' + 'mixed1_len20.pth'
    if os.path.exists(RTG_FILE):
        rtg_list = torch.load(RTG_FILE)
        print('Load rtg list:', RTG_FILE)
    else:
        print('RTG file missing')
        raise Exception('Please generate rtg list first')

    GRAPH_DIR = 'Graph_extracted_data'
    if not osp.exists(GRAPH_DIR):
        os.mkdir(GRAPH_DIR)
    #print('Eval_mode:', eval_mode)

    # GRAPH_FILE = 'Graph_extracted_data/' + 'mixed1_state' + str(state_dim) + '.pth'
    # #GRAPH_FILE = 'Graph_extracted_data/' + bench_des_in_env + '.pth'
    #
    # if os.path.exists(GRAPH_FILE):
    #     graph_state_list = torch.load(GRAPH_FILE)
    #     print('Load graph data list:', GRAPH_FILE)

    #print(graph_state_list[0])
    #concat_states = np.concatenate(graph_state_list, axis=0)
    #state_mean, state_std = np.mean(concat_states), np.std(concat_states)
    #print('state_mean', state_mean)
    #print('state_std', state_std)
    # graph_state_list_normed = []

    # for item in graph_state_list:
    #     graph_state_list_normed.append((item-env.graph_state_mean)/env.graph_state_std)
    # print('dataset normed')
    #print(graph_state_list_normed[0])
    #input()
    trainDS.transform = transforms.Compose([lambda data: addNormalizedStates(data, targetStats, meanVarTargetDict, targetVar=targetLbl)])


    # input()

    # if os.path.exists(DATA_PROCESSED_FILE):
    #     trainDS = torch.load(DATA_PROCESSED_FILE)
    #     print('Load processed dataset:', DATA_PROCESSED_FILE)
    # else:
    #     count = 0
    #     #for j in range(len(trainDS)):
    #     for j,batch in enumerate(trainDS):
    #         #data = addNormalizedStates(data, targetStats, meanVarTargetDict, trainDS_full, graph_state_list_normed, rtg_list, count, targetVar=targetLbl)
    #
    #
    #
    #
    #
    #
    #         start_idx = j * 21
    #         #data.states = torch.tensor([data.normANDgates, data.normDepth], dtype=torch.float32)
    #         #data.rtg = torch.tensor([[data.and_nodes]], dtype=torch.float32)
    #         trainDS[j].dones = torch.zeros(21)
    #         trainDS[j].dones[-1] = 1
    #         trainDS[j].timestep = torch.tensor([[0]])
    #         trainDS[j].rtg_list = torch.tensor(rtg_list[start_idx:start_idx + 20]).squeeze()
    #         trainDS[j].graph_state = torch.stack(graph_state_list[start_idx:start_idx + 20]).squeeze()
    #         trainDS[j].graph_state_pred = torch.stack(graph_state_list[start_idx + 1:start_idx + 21]).squeeze()
    #         # print(data)
    #         trainDS[j].edge_index = None
    #         trainDS[j].edge_type = None
    #         trainDS[j].node_id = None
    #         trainDS[j].node_type = None
    #         trainDS[j].num_inverted_predecessors = None
    #         # print(data)
    #         # input()
    #         for i in range(19):
    #             trainDS[j].timestep = torch.cat((trainDS[j].timestep, torch.tensor([[i]])), -1)
    #
    #         count += 1
    #     for data in trainDS:
    #         print(data)
    #         input()
    #     print('Processed dataset count:', count+1)
    #     torch.save(trainDS, DATA_PROCESSED_FILE)



    train_dl = DataLoader(trainDS, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=4)

    mode = variant['mode']
    #print('mode:', mode)
    states, traj_lens, returns = [], [], []

    traj_lens, returns = np.array(traj_lens), np.array(returns)

    #state_mean, state_std = 0, 1
    num_timesteps = sum(traj_lens)
    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)
    # def get_batch_openABC():
    #     s, a, r, d, rtg, timesteps, mask, Graph = [], [], [], [], [], [], [], []
    #     for temp, batch in enumerate(tqdm(train_dl, desc="Iteration", file=sys.stdout)):
    #
    #         # temp = batch.synVec
    #         # a = torch.zeros(20*variant['batch_size'], act_dim).scatter(1, temp.unsqueeze(-1), 1)
    #         # d = batch.dones
    #         # s = batch.states
    #         # Graph = batch
    #         # r = batch.target.unsqueeze(dim=1)
    #         # timesteps = batch.timestep
    #         # rtg = batch.rtg.unsqueeze(dim=1)
    #         temp = batch.synVec
    #         a.append(torch.zeros(20 * variant['batch_size'], act_dim).scatter(1, temp.unsqueeze(-1), 1))
    #         d.append(batch.dones)
    #         s.append(batch.states)
    #         Graph.append(batch)
    #         r.append(batch.target.unsqueeze(dim=1))
    #         timesteps = batch.timestep
    #         rtg = batch.rtg.unsqueeze(dim=1)
    #     return s, a, r, d, rtg, timesteps, mask, Graph
    def eval_episodes(target_rtg, cur_model,model_state_pred, iterative_mode, k_repeat, beam_width):
        # env.iterative_mode = iterative_mode
        # if iterative_mode:
        #     print('iterative_mode:', env.iterative_mode)
        returns, output_file_names, actions= [], [], []
        for _ in range(num_eval_episodes):
            with torch.no_grad():
                ret, output_file_name, actions = evaluate_episode_rtg(
                    env,
                    state_dim,
                    act_dim,
                    model=cur_model,
                    model_state_pred=model_state_pred,
                    max_ep_len=max_ep_len,
                    scale=scale,
                    target_return=target_rtg/scale,
                    mode=mode,
                    state_mean=env.graph_state_mean,
                    state_std=env.graph_state_std,
                    device=device,
                    using_state_predictor=using_state_predictor,
                    k_repeat=k_repeat,
                    beam_width=beam_width
                )
            if iterative_mode:
                pass
                #env.change_init_env(output_file_name)
                print('iterative_mode:', env.iterative_mode)
            returns.append(ret)
            output_file_names.append(output_file_name)
        #env.default_setting()
        return returns[-1], output_file_names[-1], actions
        # return {
        #     target_rew: np.mean(returns)
        #     # f'target_{target_rew}_return_mean': np.mean(returns)
        #     # f'target_{target_rew}_return_std': np.std(returns),
        #     # f'target_{target_rew}_length_mean': np.mean(lengths),
        #     # f'target_{target_rew}_length_std': np.std(lengths),
        # }

    # def eval_episodes(target_rew):
    #     def fn(model):
    #         returns, lengths = [], []
    #         for _ in range(num_eval_episodes):
    #             with torch.no_grad():
    #                 if model_type == 'dt':
    #                     ret, length = evaluate_episode_rtg(
    #                         env,
    #                         state_dim,
    #                         act_dim,
    #                         model,
    #                         max_ep_len=max_ep_len,
    #                         scale=scale,
    #                         target_return=target_rew/scale,
    #                         mode=mode,
    #                         state_mean=env.graph_state_mean,
    #                         state_std=env.graph_state_std,
    #                         device=device,
    #                         using_state_predictor=using_state_predictor
    #                     )
    #
    #             returns.append(ret)
    #             lengths.append(length)
    #         return {
    #             target_rew: np.mean(returns)
    #             # f'target_{target_rew}_return_mean': np.mean(returns)
    #             # f'target_{target_rew}_return_std': np.std(returns),
    #             # f'target_{target_rew}_length_mean': np.mean(lengths),
    #             # f'target_{target_rew}_length_std': np.std(lengths),
    #         }
    #     return fn

    dropout = 0.1
    model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],

            n_layer=3,
            n_head=8,
            n_inner=4*variant['embed_dim'],
            activation_function='relu',
            n_positions=1024,
            resid_pdrop=dropout,
            attn_pdrop=dropout,
    )
    model_state_pred = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=max_ep_len,
        hidden_size=variant['embed_dim'],

        n_layer=3,
        n_head=8,
        n_inner=4 * variant['embed_dim'],
        activation_function='relu',
        n_positions=1024,
        resid_pdrop=dropout,
        attn_pdrop=dropout,
    )
    #start_from_check_point = variant['start_from_check_point']
    start_from_check_point = False
    if start_from_check_point:
        temp = model.load_state_dict(
            torch.load(
                osp.join(DUMP_DIR, '2023-04-12_12-35mixed1-gcn-DT-epoch-11-train_loss-0.694_with_eval.pt')))
        # temp = model.load_state_dict(torch.load(osp.join(DUMP_DIR, '2023-03-21_22-15-gcn-DT-epoch-9-train_loss-0.143_with_eval.pt')))
        print('check_point_model_load_result:', temp)
    model = model.to(device=device)
    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    # states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)

    trainer = SequenceTrainer(
        model=model,
        model_state_pred=model_state_pred,
        optimizer=optimizer,
        batch_size=batch_size,
        #get_batch=get_batch_openABC,
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
        eval_fns=eval_episodes,
        #eval_fns=eval_episodes(tar) for tar in env_targets],  # such as [fn(1), fn(0.5), fn(0)]
        act_dim=act_dim,
        state_dim=state_dim,
        desID=test_des_in_env,
        init_node=env.init_and_nodes,
        extended_mode=extended_mode,
        state_pred_mode=using_state_predictor,
        final_repeat=final_repeat,
        beam_width=beam_width
    )

    for iter in range(variant['max_iters']):  # outer loop that including train and eval
        outputs = trainer.train_iteration(train_dl=train_dl, num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True, eval_mode=eval_mode)
    print('runtime_for_synthesis:', env.runtime_for_synthesis)
    print('runtime_for_graph_extract:', env.runtime_for_graph_extract)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--design', type=str, default='tv80')
    parser.add_argument('--design_file', type=str, default='tv80')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='mode1')  # method of norm state
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=256)  # hidden size
    parser.add_argument('--warmup_steps', type=int, default=1)
    parser.add_argument('--num_eval_episodes', type=int, default=1)  # the num of iterative eval-loops after training
    parser.add_argument('--max_iters', type=int, default=1)  # OUTER loop
    parser.add_argument('--num_steps_per_iter', type=int, default=20)  # the num of train loops before eval
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--using_state_predictor', '-up', type=bool, default=False)
    parser.add_argument('--Finetune', '-ft', type=bool, default=False)  # Finetune with current traj or random traj
    parser.add_argument('--only_eval_mode', '-ev', type=bool, default=False)
    parser.add_argument('--start_from_check_point', '-cp', type=bool, default=False)


    args = parser.parse_args()
    print(args)
    experiment('GCN+DT-experiment', variant=vars(args))
