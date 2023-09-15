import pickle
import fcntl
import numpy as np
import torch
import random
import time
from typing import List, Optional, Tuple, Dict, Union, Any
import re, os
import copy
import subprocess
from subprocess import check_output
from torch.utils.data import Dataset

import torch.nn.functional as F
from torch.distributions import Categorical
from QoR_predictor import nodes_predict

extended_mode = False


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
def make_abc_baseline_commands(actions, input_file_name, input_design_name, output_file_name=None): # do not use scl
    # abc_command = "read stdcells.lib; "  # this file is standard cell library
    #abc_command = "read asap7.lib; "  # this file is standard cell library
    abc_command = ""
    abc_command += "read " + input_file_name + "; "
    abc_command += "strash; "  # convert the input file into aig type
    for i in actions:
        abc_command += takeAction3(i) + "; "
        #print(takeAction2(i))
    if output_file_name is not None:
        abc_command += " write " + output_file_name + "; "
        #print('is not none')
    else:
        abc_command += ""
        #print('is none')

    #abc_command += 'write_bench -l ' + input_design_name + '.bench; '
    abc_command += 'print_stats'
    return abc_command
def make_abc_commands(actions, input_file_name, input_design_name, extended_mode, output_file_name=None): # do not use scl

    action_ = actions[-1:]

    abc_command = ""
    abc_command += "read " + input_file_name + "; "
    abc_command += "strash; "  # convert the input file into aig type

    for i in action_:
        abc_command += takeAction2(i) + "; "
    if extended_mode:
        for i in actions:
            abc_command += takeAction2(i) + "; "
    #abc_command += 'write_bench -l ' + input_design_name + '.bench; '
    #bench_file_path =
    # if len(actions) > 0:
    #     bench_file_path = input_file_name[:-6] + str(actions[-1]) + '.bench'
    #     if os.path.exists(bench_file_path):
    #         pass
    #     else:
    #         abc_command += 'write_bench -l ' + output_file_name_temp
    # else:
    #     bench_file_path = input_file_name[:-6] + '.bench'
    #     if os.path.exists(bench_file_path):
    #         pass
    #     else:
    #         abc_command += 'write_bench -l ' + output_file_name_temp
    if not os.path.exists(output_file_name):
        abc_command += 'write_bench -l ' + output_file_name
    abc_command += '; print_stats'
    return abc_command

def make_final_abc_commands(actions, input_file_name, input_design_name, output_file_name=None): # do not use scl
    action = actions

    abc_command = ""
    abc_command += "read " + input_file_name + "; "
    abc_command += "strash; "  # convert the input file into aig type

    for i in action:
        abc_command += takeAction2(i) + "; "

    #abc_command += 'write_bench -l ' + input_design_name + '.bench; '
    #bench_file_path =
    # if len(actions) > 0:
    #     bench_file_path = input_file_name[:-6] + str(actions[-1]) + '.bench'
    #     if os.path.exists(bench_file_path):
    #         pass
    #     else:
    #         abc_command += 'write_bench -l ' + output_file_name_temp
    # else:
    #     bench_file_path = input_file_name[:-6] + '.bench'
    #     if os.path.exists(bench_file_path):
    #         pass
    #     else:
    #         abc_command += 'write_bench -l ' + output_file_name_temp
    abc_command += 'write_bench -l ' + output_file_name
    abc_command += '; print_stats'
    return abc_command
def get_metrics(stats) -> Dict[str, Union[float, int]]:
    """
    parse LUT count and levels from the stats command of ABC
    """
    line = stats.decode("utf-8").split('\n')[-2].split(':')[-1].strip()
    #print('line:', line)
    results = {}
    ob = re.search(r'lev *= *[0-9]+', line)
    results['levels'] = int(ob.group().split('=')[1].strip())

    ob = re.search(r'nd *= *[0-9]+', line)
    results['lut'] = int(ob.group().split('=')[1].strip())

    ob = re.search(r'i/o *= *[0-9]+ */ *[0-9]+', line)
    results['input_pins'] = int(ob.group().split('=')[1].strip().split('/')[0].strip())
    results['output_pins'] = int(ob.group().split('=')[1].strip().split('/')[1].strip())

    ob = re.search(r'edge *= *[0-9]+', line)
    results['edges'] = int(ob.group().split('=')[1].strip())

    ob = re.search(r'lat *= *[0-9]+', line)
    results['latches'] = int(ob.group().split('=')[1].strip())

    return results

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
def aig_evaluate(design_file: str, sequence: List[Union[str, int]],  use_yosys: bool,
                  compute_init_stats: bool = False, verbose: bool = False,
                  write_unmap_design_path: Optional[str] = None) \
        -> Tuple[int, int, Dict[str, Any]]:
    """
         Get property of the design after applying sequence of operations `sequence`

        Args:
            design_file: path to the design 'path/to/design.blif'
            sequence: sequence of operations
                        -> either identified by id number
                            0: rewrite
                            1: rewrite -z...
                        -> or by operation names
                            `rewrite`
                            `rewrite -z`
                            ...
            lut_inputs: number of LUT inputs (2 < num < 33)
            use_yosys: whether to use yosys-abc or abc_py
            compute_init_stats: whether to compute and store initial stats on delay and area
            verbose: verbosity level
        Returns:
            lut_k, level and extra_info (execution time, initial stats)
        Exception: CalledProcessError
    """

    assert not compute_init_stats
    t_ref = time.time()
    extra_info: Dict[str, Any] = {}

    if sequence is None:
        sequence = []
    sequence = ['strash; '] + sequence
    #abc_command = 'read ' + self.params['mapping']['library_file'] + '; '
    #abc_command = 'read ' + 'asap7.lib' + '; '
    abc_command = ""
    abc_command += 'read ' + design_file + '; '
    abc_command += ';'.join(sequence) + '; '
    if write_unmap_design_path is not None:
        abc_command += 'write ' + write_unmap_design_path + '; '

    #abc_command += 'map -D 20000000;'

    #abc_command += 'topo; stime;'
    abc_command += 'print_stats;'

    #abc_command += f"if {'-v ' if verbose > 0 else ''}-K {lut_inputs}; "
    #abc_command += 'print_stats; '
    print('abc_command:', abc_command)
    cmd_elements = ['yosys-abc', '-c', abc_command]
    #print('cmd_elements:', cmd_elements)
    proc = subprocess.check_output(cmd_elements)
    # read results and extract information
    print('proc:', proc)
    line = proc.decode("utf-8").split('\n')[-2].split(':')[-1].strip()
    #print('line:', line)

    resultName = str(design_file) + "-basic-info.csv"
    #a/=0

    print('abc_command:', abc_command)
    print('line:', line)
    ob = re.search(r'lev *= *[0-9]+', line)
    #print('ob:', ob)
    if ob is None:
        print("----" * 10)
        print(f'Command: {" ".join(cmd_elements)}')
        print(f"Out line: {line}")
        print(f"Design: {design_file}")
        print(f"Sequence: {sequence}")
        print("----" * 10)
    levels = int(ob.group().split('=')[1].strip())

    ob = re.search(r'and *= *[0-9]+', line)
    NumAnd = int(ob.group().split('=')[1].strip())

    extra_info['exec_time'] = time.time() - t_ref
    print('NumAnd:', NumAnd)
    print('Lev:', levels)
    # with open(resultName, 'a') as andLog:
    #     line = ""
    #     line += str(NumAnd)
    #     line += " "
    #     line += str(levels)
    #     line += " "
    #     line += str(extra_info['exec_time'])
    #     line += " "
    #     line += str(time.time())
    #     line += "\n"
    #     andLog.write(line)
    return NumAnd, levels, extra_info
def evaluate_episode(  # useless
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=20,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    #model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    env.reset()
    state = torch.tensor([[env.init_and_nodes, env.init_depth]], dtype=torch.float32)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        print(states)
        print(actions)
        print(rewards)
        print(target_return)
        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length

class CustomDataset(Dataset):
    def __init__(self):
        self.data = {}
    def __getitem__(self, index):
        #state, action, reward = self.data[index]
        return self.data
    def __len__(self):
        return len(self.data)
    def add_data(self, state, action, reward):
        #self.data.append((state, action, reward))
        self.data['graph_state'] = state
        self.data['rtg_list'] = reward
        self.data['synVec'] = action

def evaluate_episode_rtg(  # model = dt
        env,
        state_dim,
        act_dim,
        model,
        model_state_pred,
        max_ep_len=20,
        scale=1.,
        state_mean=None,
        state_std=None,
        device='cuda',
        target_return=None,
        mode='normal',
        using_state_predictor=True,
        extended_mode=False,
        k_repeat=1,
        beam_width=1
    ):

    #torch.no_grad()
    #print('Target return:', target_return)
    model.eval()
    model.to(device=device)
    using_state_predict_model = False
    state = env.reset()
    # cur_graph_mean = env.graph_state_mean
    # cur_graph_std = env.graph_state_std
    #print('Init_state:', state)


    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    # states = state.reshape(1, state_dim).to(device=device, dtype=torch.float32)
    # actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    # action_indexes = []
    #print('actions:', actions)
    # rewards = torch.ones(0, device=device, dtype=torch.float32)

    ep_return = target_return
    # target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    # timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0, 0

    beam = [{'beam_states': state.reshape(1, state_dim).to(device=device, dtype=torch.float32),
             'beam_actions': torch.zeros((0, act_dim), device=device, dtype=torch.float32),
             'beam_rewards': torch.ones(0, device=device, dtype=torch.float32),
             'beam_sum_reward': 0.,
             'beam_product_chance': 1.,
             'beam_action_idxes': [],
             'beam_target_return': torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1),
             'timesteps': torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1),
             'score': None}]  # candidate

    #max_ep_len = 50
    for t in range(max_ep_len):
        current_beam_candidate_size = len(beam)
        #print('current_beam_candidate_size:', current_beam_candidate_size)
        for i in range(current_beam_candidate_size):
            beam[i]['beam_actions'] = torch.cat([beam[i]['beam_actions'], torch.zeros((1, act_dim), device=device)], dim=0)
            beam[i]['beam_rewards'] = torch.cat([beam[i]['beam_rewards'], torch.ones(1, device=device)])
            #mean_ = env.mean_state
            mean_ = 0
            std_ = 1.

            action, state_pred, reward_pred = model.get_action(
                #(beam[i]['beam_states'].to(dtype=torch.float32) - mean_)/std_,
                beam[i]['beam_states'].to(dtype=torch.float32),
                beam[i]['beam_actions'].to(dtype=torch.float32),
                beam[i]['beam_rewards'].to(dtype=torch.float32),
                beam[i]['beam_target_return'].to(dtype=torch.float32),
                beam[i]['timesteps'].to(dtype=torch.long),
            )
            _, state_pred, _ = model_state_pred.get_action(
                # (beam[i]['beam_states'].to(dtype=torch.float32) - mean_)/std_,
                beam[i]['beam_states'].to(dtype=torch.float32),
                beam[i]['beam_actions'].to(dtype=torch.float32),
                beam[i]['beam_rewards'].to(dtype=torch.float32),
                beam[i]['beam_target_return'].to(dtype=torch.float32),
                beam[i]['timesteps'].to(dtype=torch.long),
            )
            topk_actions, topk_idx = torch.topk(action, k=beam_width)

            for k in range(beam_width - 1):
                beam.append(copy.deepcopy(beam[i]))
            #print(beam[i]['beam_action_idxes'])
            for j in range(beam_width):  # for each beam, produce offsprings after choosing topk actions
                #print(beam[i + j]['beam_actions'])
                if j == 0:
                    beam_idx = i
                else:
                    beam_idx = -j

                action_index = topk_idx[j].detach().numpy()
                beam[beam_idx]['beam_action_idxes'].append(action_index.tolist())
                #print(beam[beam_idx]['beam_action_idxes'])
                beam[beam_idx]['beam_actions'][-1] = torch.zeros(1, act_dim).to(device).scatter(1,
                                                         torch.tensor([int(action_index)]).to(device).unsqueeze(
                                                            -1), 1)
                action_str = ''
                for act in beam[beam_idx]['beam_action_idxes']:
                    action_str += str(act)

                # if extended_mode:
                #     #input()
                #     pkl_file_name = "playground_extend/" + str(env.desID) + "/" + str(env.desID) + "-" +action_str + ".pkl"
                # elif using_state_predictor:
                #     pkl_file_name = "playground_state_predict/" + str(env.desID) + "/" + str(env.desID) + "-" + action_str + ".pkl"
                # else:
                #     pkl_file_name = "playground/" + str(env.desID) + "/" + str(env.desID) + "-" +action_str + ".pkl"
                pkl_file_name = "playground/" + str(env.desID) + "/" + str(env.desID) + "-" + action_str + ".pkl"

                if os.path.exists(pkl_file_name):  # check if there exist cache
                    with open(pkl_file_name, 'rb+') as f1:
                        fcntl.flock(f1, fcntl.LOCK_EX)
                        state, reward, done, sum_reward = pickle.load(f1)
                        fcntl.flock(f1, fcntl.LOCK_UN)
                else:
                    if using_state_predictor:  # get rid of simulator
                        #state = torch.round(state_pred * 1e12 * 0.5)/1e12 + env.reset() * 0.5
                        #state = torch.round(state_pred.clone().detach() * 1e9) / 1e9
                        #state = env.reset()
                        #print('max:', torch.max(state).item(), end=' ')
                        #print(state_pred)
                        state = state_pred
                        #state =
                        #reward = reward_pred
                        done = False
                        temp = beam[beam_idx]['beam_sum_reward']
                        sum_reward = (1 - nodes_predict(env.qor_predict_model, env.desID, beam[beam_idx]['beam_action_idxes'])) * 0.5 * env.rtg_baseline * len(beam[beam_idx]['beam_action_idxes']) *0.05
                        print('pred_sum_of_reward:', sum_reward)
                        #print('pred_qor:', -nodes_predict(env.qor_predict_model, env.desID, beam[beam_idx]['beam_action_idxes']))
                        #sum_reward = 0
                        reward = sum_reward - temp
                        #print('reward', reward)
                        #reward = reward_pred
                        #reward = 0

                    else:
                        #input()
                        #print(beam[beam_idx]['beam_action_idxes'])
                        if len(beam[beam_idx]['beam_action_idxes']) == 30:
                            state, reward, done, sum_reward = env.step(beam[beam_idx]['beam_action_idxes'], skip_graph_extraction=False)
                        else:
                            state, reward, done, sum_reward = env.step(beam[beam_idx]['beam_action_idxes'],
                                                                       skip_graph_extraction=False)
                            #state = state_pred
                        #print(sum_reward)
                        #input()
                        using_ground_truth = True
                        # if using_state_predict_model:

                        # else:
                        #     state = env.reset()
                        #print('max:', torch.max(state).item(), end=' ')
                    # f2 = open(pkl_file_name, 'wb')
                    # fcntl.flock(f2, fcntl.LOCK_EX)
                    # pickle.dump([state, reward, done, sum_reward], f2)
                    # f2.close()
                    # fcntl.flock(f2, fcntl.LOCK_UN)
                    with open(pkl_file_name, 'wb+') as f2:
                        fcntl.flock(f2, fcntl.LOCK_EX)
                        pickle.dump([state, reward, done, sum_reward], f2)
                        fcntl.flock(f2, fcntl.LOCK_UN)
                #state, reward, done, sum_reward = env.step(beam[beam_idx]['beam_action_idxes'])
                cur_state = state.to(device=device).reshape(1, state_dim)
                beam[beam_idx]['beam_states'] = torch.cat([beam[beam_idx]['beam_states'], cur_state], dim=0)
                beam[beam_idx]['beam_rewards'][-1] = reward
                beam[beam_idx]['beam_sum_reward'] = sum_reward
                pred_return_legacy = beam[beam_idx]['beam_target_return'][0, -1] - (reward / scale)
                #print("pred_return_legacy:", pred_return_legacy)
                #print("target_return:", target_return)

                #pred_return = torch.tensor(ep_return - sum_reward)
                #print("pred_return:", pred_return)

                beam[beam_idx]['beam_target_return'] = torch.cat([beam[beam_idx]['beam_target_return'], pred_return_legacy.reshape(1, 1)], dim=1)
                if t > 21:
                    pass
                else:
                    beam[beam_idx]['timesteps'] = torch.cat(
                        [beam[beam_idx]['timesteps'],
                         torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

        current_beam_size = len(beam)

        if current_beam_size > beam_width:

            #print('Shrinking')
            beam_sorted = sorted(beam, key=lambda x: x['beam_sum_reward'], reverse=True)
            print_log = False
            if print_log:
                for item in beam:
                    print(item['beam_action_idxes'])
                    print(item['beam_sum_reward'])
            #beam_sorted = sorted(beam, key=lambda x: x['beam_product_chance'], reverse=True)
            beam = beam_sorted[:beam_width]
            # print('After Shrinking')

        #use_state_predictor = False
        '''if using_state_predictor:
            if t < max_ep_len - 1:
                state = state_pred
                reward = reward_pred
                #reward = 10
                # print('reward_pred:', reward_pred)
                done = False
                env.actions.append(action_index)
            else:
                done = True
                env.step(action_index)
                # input()
        else:
            state, reward, done = env.step(action_index)  # origin
            #print('state_pred:', state_pred)
            #print('state:', state)
            #print('state_euclidean_dis:', torch.norm(state.squeeze()-state_pred))
            #print('state_cosine_similarity:', torch.dot(state.squeeze(), state_pred)/(torch.norm(state.squeeze()) * torch.norm(state_pred)))
            #print('reward_pred:', reward_pred)
            #print('reward:', reward)
            #print('reward_euclidean_dis:', torch.norm(reward - reward_pred))'''

        # state, reward, done = env.step(action_index)  # origin

        # cur_graph_mean = torch.mean(state)
        # cur_graph_std = torch.std(state)
        # state_means = []
        # state_stds = []
        # state_means.append(torch.mean(state))
        # state_stds.append(torch.std(state))

        #print('state_mean:', torch.mean(state))
        #print('state_std:', torch.std(state))
        #cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        # cur_state = state.to(device=device).reshape(1, state_dim)
        # states = torch.cat([states, cur_state], dim=0)
        # rewards[-1] = reward

        # pred_return = target_return[0,-1] - (reward/scale)
        # if ...
        # else:
        #     pred_return = target_return[0,-1]
        # target_return = torch.cat(
        #     [target_return, pred_return.reshape(1, 1)], dim=1)

        #print('time_step_shape:', timesteps.shape)
        #print(timesteps)

        episode_return += 0

        # episode_return = (env.init_and_nodes - env.and_nodes)/env.init_and_nodes
        #print('episode_return:', episode_return)
        #print('episode_length:', episode_length)
        episode_length += 1

        # if done:
        #     target_return_list = target_return[-1].tolist()
        #     target_return_list = [np.round(x, 4) for x in target_return_list]
        #     init_rtg = target_return_list[0]
        #     print('\n')
        #     #print(target_return_list)
        #     #print([init_rtg-i for i in target_return_list])
        #     print(action_indexes)
        #     print('-' * 40)
        #     #print('\n')
        #     #print(env.actions)
        #     break

    # finetune_dataset = CustomDataset()
    # rtgs = env.rtg_baseline - rewards
    # finetune_dataset.add_data(states, actions, rtgs)
    # ft_dataloader = DataLoader(finetune_dataset, batch_size=1, shuffle=True)
    #
    # print(finetune_dataset)
    returns = []
    beam_sorted = sorted(beam, key=lambda x: x['beam_sum_reward'], reverse=True)
    # beam_sorted = sorted(beam, key=lambda x: x['beam_product_chance'], reverse=True)
    beam_ = beam_sorted[:1]
    using_k_repeat = False  # 25 repeats
    k = 1
    if using_state_predictor or k_repeat > 1:
        using_k_repeat = True
        k = k_repeat
    if extended_mode:
        using_k_repeat = False
    output_file_name = None
    print_info = True
    action_str = ''
    #print('length of beam:', len(beam))
    if using_k_repeat:  # if k=1, return true imprv
        print('Using k-repeat')
        for item in beam:
            #returns.append(item['beam_sum_reward'])
            #print(item['beam_sum_reward'])
            if print_info:
                print(k, 'repeat:', item['beam_action_idxes'], end=' ')

            action_str = ''
            for act in item['beam_action_idxes']:
                action_str += str(act)

            if using_state_predictor:
                playground = "playground_state_predict/"
            else:
                playground = "playground/"
            pkl_file_name = playground + str(env.desID) + "/" + str(env.desID) + "-" + action_str + "-repeat" + str(k) + "-final.pkl"
            output_file_name = playground + str(env.desID) + "/" + str(env.desID) + "-25repeats.bench"
            if os.path.exists(pkl_file_name) and not env.iterative_mode:  # check if there exist cache
            #if False:  # check if there exist cache
                with open(pkl_file_name, 'rb+') as f1:
                    fcntl.flock(f1, fcntl.LOCK_EX)
                    Lev, NumAnd = pickle.load(f1)
                    fcntl.flock(f1, fcntl.LOCK_UN)
                if print_info:
                    print('')
            else:
                k_repeat_runs = []
                for k in range(k):
                    k_repeat_runs.extend(item['beam_action_idxes'])
                # print('length of seq:', len(k_repeat_runs))
                #output_file_name = "playground/" + str(env.desID) + "/" + str(env.desID) + "-25repeats.bench"
                abc_command = make_final_abc_commands(k_repeat_runs, env.des_file, env.desID,
                                                      output_file_name=output_file_name)
                #print('abc_command:', abc_command)
                start_syn = time.time()
                proc = check_output(['yosys-abc', '-c', abc_command])
                Lev, NumAnd = get_metrics2(proc)
                end_syn = time.time()
                during = end_syn - start_syn
                env.runtime_for_synthesis += during
                # f2 = open(pkl_file_name, 'wb')
                # fcntl.flock(f2, fcntl.LOCK_EX)
                # pickle.dump([Lev, NumAnd], f2)
                # f2.close()
                # fcntl.flock(f2, fcntl.LOCK_UN)
                with open(pkl_file_name, 'wb+') as f3:
                    fcntl.flock(f3, fcntl.LOCK_EX)
                    pickle.dump([Lev, NumAnd], f3)
                    fcntl.flock(f3, fcntl.LOCK_UN)
                if print_info:
                    print('imprv:', (env.init_and_nodes - NumAnd) / env.init_and_nodes)
            #print("Final NumAnd", NumAnd)
            # if env.iterative_mode:
            #     print('Reload init state tensor')
            #     env.change_init_env(output_file_name)

            sum_reward = (env.init_and_nodes - NumAnd) / env.init_and_nodes
            returns.append(sum_reward)
    else:
        for item in beam_:
            if print_info:
                print('Target return:', '{:.4f}'.format(target_return), end=' ')
                print(item['beam_action_idxes'], end=' ')
                print('Imprv:', item['beam_sum_reward'])
                #print('\n')
            returns.append(item['beam_sum_reward'])
    episode_return = max(returns)
    #print('imprv:', episode_return)
    #print('beam:', beam[0]['beam_action_idxes'])
    return max(returns), output_file_name, beam_[0]['beam_action_idxes']
    #return beam_[0]['beam_sum_reward'], output_file_name, beam_[0]['beam_action_idxes']
    #return episode_return

if __name__ == '__main__':
    use_yosys=True
    actions = [1, 2, 3, 4, 0]
    #sequence = [(action_space[ind].act_id if not use_yosys else action_space[ind].act_str) for ind in sequence]
    abc_command = make_abc_commands(actions, 'tv80.aig')
    print('abc_command:', abc_command)
    proc = check_output(['yosys-abc', '-c', abc_command])
    print('proc:', proc)
    Lev, NumAnd = get_metrics2(proc)
    print('Lev:', Lev)
    print('NumAnd:', NumAnd)
