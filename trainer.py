import os

import numpy as np
import torch
import os.path as osp
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import random
import pickle
from torch_geometric.data import Dataset, DataLoader
from evaluate_episodes import CustomDataset
from bayes_opt import BayesianOptimization
from joblib import Parallel, delayed

def prRed(prt): print("\033[91m {}\033[00m".format(prt))
def prGreen(prt): print("\033[92m {}\033[00m".format(prt))

def prYellow(prt): print("\033[93m {}\033[00m".format(prt))

def prLightPurple(prt): print("\033[94m {}\033[00m".format(prt))

def prPurple(prt): print("\033[95m {}\033[00m".format(prt))

def prCyan(prt): print("\033[96m {}\033[00m".format(prt))

def prLightGray(prt): print("\033[97m {}\033[00m".format(prt))

def prBlack(prt): print("\033[98m {}\033[00m".format(prt))


DUMP_DIR = "/home/lcy/PycharmProjects/OpenABC/SynthNETV5_set1-test"
device = 'cpu'


def plotScatter(x, y, xlabel, ylabel, title, extended_mode, state_pred, beam_width):
    #fig = plt.figure(figsize=(10, 6))
    #ax = fig.add_subplot(1, 1, 1)

    plt.scatter(x, y, s=1.5)
    #leg = plt.legend(loc='best', ncol=2, shadow=True, fancybox=True)
    #leg.get_frame().set_alpha(0.5)
    plt.xlabel(xlabel, weight='bold')
    plt.ylabel(ylabel, weight='bold')
    plt.title(title, weight='bold')
    time_str = time.strftime('%Y-%m-%d_%H-%M', time.localtime())
    suffix = ''
    repeat_mode = False
    iterative_mode = False
    extend_mode = extended_mode
    #beam_mode = True
    fine_grit = False
    broad_search = False  # large rtg scale
    if repeat_mode:
        suffix += '-25repeat4top'
    if iterative_mode:
        suffix += '-25iteration'
    if extend_mode:
        suffix += '-extend'
    if beam_width > 1:
        suffix += '-beam'
        suffix += str(beam_width)
    if fine_grit:
        suffix += '-FineGrained'
    if broad_search:
        suffix += '-Broad'
    if state_pred:
        suffix += '-StatePredict'
    else:
        suffix += '-GroundTruth'
    #suffix += '-ShiftedMean'

    #plt.savefig('rtg_scatter-0.482/' + str(time_str) + '-' + title + suffix + '.pdf', bbox_inches='tight')
    #plt.savefig('rtg_scatter-0.482/' + str(time_str) + '-' + title + suffix + '.png', bbox_inches='tight')
    #plt.savefig('rtg_scatter-0.420/' + str(time_str) + '-' + title + suffix + '.pdf', bbox_inches='tight')
    #plt.savefig('rtg_scatter-0.420/' + str(time_str) + '-' + title + suffix + '.png', bbox_inches='tight')
    #plt.savefig('rtg_scatter-0.386/' + str(time_str) + '-' + title + suffix + '.pdf', bbox_inches='tight')
    #plt.savefig('rtg_scatter-0.386/' + str(time_str) + '-' + title + suffix + '.png', bbox_inches='tight')
    #plt.savefig('rtg_scatter-0.383/' + str(time_str) + '-' + title + suffix + '.pdf', bbox_inches='tight')
    #plt.savefig('rtg_scatter-0.383/' + str(time_str) + '-' + title + suffix + '.png', bbox_inches='tight')
    #plt.savefig('rtg_scatter-0.381/' + str(time_str) + '-' + title + suffix + '.pdf', bbox_inches='tight')
    #plt.savefig('rtg_scatter-0.381/' + str(time_str) + '-' + title + suffix + '.png', bbox_inches='tight')
    plt.savefig('rtg_scatter/' + str(time_str) + '-' + title + suffix + '.pdf', bbox_inches='tight')
    plt.savefig('rtg_scatter/' + str(time_str) + '-' + title + suffix + '.png', bbox_inches='tight')
    #plt.savefig('rtg_scatter-0.410/' + str(time_str) + '-' + title + suffix + '.pdf', bbox_inches='tight')
    #plt.savefig('rtg_scatter-0.410/' + str(time_str) + '-' + title + suffix + '.png', bbox_inches='tight')
    #plt.savefig('rtg_scatter-0.299/' + str(time_str) + '-' + title + suffix + '.pdf', bbox_inches='tight')
    #plt.savefig('rtg_scatter-0.299/' + str(time_str) + '-' + title + suffix + '.png', bbox_inches='tight')
    pkl_file_name = 'rtg_scatter/' + str(time_str) + '-' + title + suffix + '.pkl'
    with open(pkl_file_name, 'wb+') as f2:
        pickle.dump([x, y], f2)

# def rtg2imprv(rtg):
#     # return np.sin(10 * np.pi * x) / (2 * x) + (x - 1) ** 4
#     #outputs = eval_fns(rtg, model)
#     # self.collected_datasets.append(colleted_dataset)
#     # print('outputs:', outputs)
#     #return outputs
#     return rtg
class Trainer:
    def __init__(self, model, model_state_pred, optimizer, batch_size, loss_fn, act_dim, state_dim, desID, init_node, extended_mode,state_pred_mode, final_repeat, beam_width, scheduler=None, eval_fns=None):
        self.model = model
        self.model_state_pred = model_state_pred
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.start_time = time.time()
        self.desID = desID
        self.init_node = init_node
        self.collected_datasets = []
        self.extended_mode = extended_mode
        self.state_pred_mode = state_pred_mode
        self.final_repeat = final_repeat
        self.beam_width = beam_width
        self.best_actions = []
        self.best_imprv = 0.
        self.best_rtg = 1.


    def rtg2imprv(self, rtg, k_repeat=1):
        iterative_mode = False
        outputs, output_file, actions = self.eval_fns(rtg, self.model, self.model_state_pred, iterative_mode, k_repeat, self.beam_width)
        # self.collected_datasets.append(colleted_dataset)

        if outputs > self.best_imprv:
            #print(actions)
            self.best_imprv = outputs
            self.best_actions = actions
            self.best_rtg = rtg
        # print(actions)
        # print('outputs:', outputs)
        # print('self.best_imprv:', self.best_imprv)

        return [outputs, actions]
    def train_iteration(self, train_dl, num_steps, iter_num=0, print_logs=True, eval_mode=False):  # outer loop
        train_losses = []
        logs = dict()
        train_start = time.time()
        valid_curve = []
        train_losses = []
        trainLossOpt = 0
        bestTrainEpoch = 1
        DUMP_DIR = "/home/lcy/PycharmProjects/OpenABC/SynthNETV5_set1-test"
        if not eval_mode:
            for ep in range(num_steps):  # num of rounds for training before eval
                self.model.train()
                episode_losses = []
                for temp, batch in enumerate(tqdm(train_dl, desc="Iteration", file=sys.stdout)):
                    batch = batch.to(device)
                    train_loss = self.train_step(batch)
                    train_losses.append(train_loss)
                    episode_losses.append(train_loss)
                    if self.scheduler is not None:
                        self.scheduler.step()
                    #print({'Train loss': train_loss})
                print({'Episode losses': sum(episode_losses)/len(episode_losses)})
                episode_losses.clear()
                validLossOpt = train_losses[-1]
                time_str = time.strftime('%Y-%m-%d_%H-%M', time.localtime())
                torch.save(self.model.state_dict(),
                           osp.join(DUMP_DIR,
                                    time_str + self.desID + '-gcn-DT-epoch-{}-train_loss-{:.3f}.pt'.format(ep, validLossOpt)))
                torch.save(self.model,
                           osp.join(DUMP_DIR,
                                    time_str + self.desID + '-gcn-DT-epoch-{}-train_loss-{:.3f}.pth'.format(ep, validLossOpt)))

                self.model.eval()  # frezeeon para before save

                torch.save(self.model.state_dict(),
                           osp.join(DUMP_DIR,
                                    time_str + self.desID + '-gcn-DT-epoch-{}-train_loss-{:.3f}_with_eval.pt'.format(ep, validLossOpt)))
                torch.save(self.model,
                           osp.join(DUMP_DIR,
                                    time_str + self.desID + '-gcn-DT-epoch-{}-train_loss-{:.3f}_with_eval.pth'.format(ep, validLossOpt)))
                self.model.train()

            #plotChart([i + 1 for i in range(len(train_losses))], train_losses, "# Epochs", "Loss", "train_loss","Training loss")
            # Loading best validation model
            #self.model.load_state_dict(torch.load(osp.join(DUMP_DIR, 'gcn-epoch-{}-val_loss-{:.3f}.pt'.format(bestTrainEpoch, trainLossOpt))))
            #print({'Train losses': train_losses[-1]})
        logs['time/training'] = time.time() - train_start

        #self.model.eval()
        if eval_mode:  # load previous model
            load_DTLS = self.model.load_state_dict(
                #torch.load(osp.join(DUMP_DIR, '2023-04-06_15-12tv80-gcn-DT-epoch-29-train_loss-0.284_with_eval.pt')))
                #torch.load(osp.join(DUMP_DIR, '2023-04-06_16-37tv80-gcn-DT-epoch-29-train_loss-0.118_with_eval.pt')))
                #torch.load(osp.join(DUMP_DIR, '2023-04-18_12-07mixed1-gcn-DT-epoch-19-train_loss-0.588_with_eval.pt')))
                #torch.load(osp.join(DUMP_DIR, '2023-04-18_11-10mixed1-gcn-DT-epoch-7-train_loss-0.526_with_eval.pt')))
                #torch.load(osp.join(DUMP_DIR, '2023-04-30_00-57mixed1-gcn-DT-epoch-11-train_loss-1.312_with_eval.pt')))
                #torch.load(osp.join(DUMP_DIR, '2023-05-01_10-53mixed1-gcn-DT-epoch-9-train_loss-0.964_with_eval.pt')))
                #torch.load(osp.join(DUMP_DIR, '2023-05-02_14-32mixed1-gcn-DT-epoch-11-train_loss-0.528_with_eval.pt')))
                #torch.load(osp.join(DUMP_DIR, '2023-05-02_17-42mixed1-gcn-DT-epoch-17-train_loss-0.482_with_eval.pt')))
                torch.load(osp.join(DUMP_DIR, '2023-05-04_15-17mixed1-gcn-DT-epoch-7-train_loss-0.420_with_eval.pt')))
                #torch.load(osp.join(DUMP_DIR, '2023-05-19_15-04mixed1-gcn-DT-epoch-15-train_loss-0.504_with_eval.pt')))
                #torch.load(osp.join(DUMP_DIR, '2023-05-21_08-09mixed1-gcn-DT-epoch-3-train_loss-0.431_with_eval.pt')))
                #torch.load(osp.join(DUMP_DIR, '2023-05-18_09-16mixed1-gcn-DT-epoch-17-train_loss-0.575_with_eval.pt')))
                #torch.load(osp.join(DUMP_DIR, '2023-05-17_13-14mixed1-gcn-DT-epoch-18-train_loss-0.705_with_eval.pt')))
                #torch.load(osp.join(DUMP_DIR, '2023-05-17_09-43mixed1-gcn-DT-epoch-11-train_loss-0.783_with_eval.pt')))
                #torch.load(osp.join(DUMP_DIR, '2023-05-15_19-37mixed1-gcn-DT-epoch-17-train_loss-2.051_with_eval.pt')))
                #torch.load(osp.join(DUMP_DIR, '2023-05-15_15-35mixed1-gcn-DT-epoch-9-train_loss-2.095_with_eval.pt')))
                #torch.load(osp.join(DUMP_DIR, '2023-05-14_05-40mixed1-gcn-DT-epoch-7-train_loss-0.381_with_eval.pt')))
                #torch.load(osp.join(DUMP_DIR, '2023-05-12_14-19mixed1-gcn-DT-epoch-10-train_loss-0.406_with_eval.pt')))
                #torch.load(osp.join(DUMP_DIR, '2023-05-10_15-26mixed1-gcn-DT-epoch-13-train_loss-0.315_with_eval.pt')))
                #torch.load(osp.join(DUMP_DIR, '2023-05-10_21-09mixed1-gcn-DT-epoch-16-train_loss-0.299_with_eval.pt')))
                #torch.load(osp.join(DUMP_DIR, '2023-05-12_09-50mixed1-gcn-DT-epoch-1-train_loss-0.410_with_eval.pt')))
                #torch.load(osp.join(DUMP_DIR, '2023-05-13_14-43mixed1-gcn-DT-epoch-18-train_loss-0.386_with_eval.pt')))
                #torch.load(osp.join(DUMP_DIR, '2023-05-13_21-26mixed1-gcn-DT-epoch-11-train_loss-0.383_with_eval.pt')))
            load_graph_state_pred = self.model_state_pred.load_state_dict(
                torch.load(osp.join(DUMP_DIR, '2023-05-19_15-04mixed1-gcn-DT-epoch-15-train_loss-0.504_with_eval.pt')))
            # temp = self.model.load_state_dict(
            #     torch.load(osp.join(DUMP_DIR, 'gcn-epoch-{}-val_loss-{:.3f}.pt'.format(1, 0.126))))
            print('model_load_result:', load_DTLS)
            print('graph_state_pred_model_load_result:', load_graph_state_pred)
            #self.model.train()
            self.model.eval()  # only this code can fix the result
            self.model_state_pred.eval()
            #for m in self.model.modules():  # close BN
                #if isinstance(m, torch.nn.Dropout):  # isinstance(m, torch.nn.BatchNorm2d) or
                #if isinstance(m, torch.nn.BatchNorm2d):  # isinstance(m, torch.nn.BatchNorm2d) or
                    #pass
                    #m.train()
                    #m.eval()

        eval_start = time.time()


        def rtg2imprv_25(rtg, k_repeat=25):
            # return np.sin(10 * np.pi * x) / (2 * x) + (x - 1) ** 4
            #time.sleep(rtg*10%5)
            iterative_mode = False
            outputs, output_file = self.eval_fns(rtg, self.model, iterative_mode, k_repeat)
            #self.collected_datasets.append(colleted_dataset)
            #print('outputs:', outputs)
            return outputs
        def bo_method(function):
            pbounds = {'rtg': (-10, 10)}
            optimizer = BayesianOptimization(
                f=function,
                pbounds=pbounds,
                random_state=1,
            )
            optimizer.maximize(
                init_points=10,
                n_iter=20,
            )
            best_rtg = optimizer.max['params']
            best_imprv = optimizer.max['target']
            return best_rtg, best_imprv

        using_BO = False
        if not using_BO:
            #rtg_list = [10, 7.5, 5, 3, 2, 1, 0.5, 0.3, 0, -1, -2, -5]
            resolution = 0.1
            lower_bound = -5
            upper_bound = 5

            rtg_list = [i*resolution for i in range(int(lower_bound/resolution), int(upper_bound/resolution))]
            #rtg_list = [i/10 for i in range(1)]
            random.seed(0)
            random.shuffle(rtg_list)
            # print('rtg_list', rtg_list)

            output_list = Parallel(n_jobs=4)(delayed(self.rtg2imprv)(i) for i in rtg_list)
            #print(output_list)
            imprv_list = [item[0] for item in output_list]
            action_list = [item[1] for item in output_list]
            self.best_imprv = max(imprv_list)
            #print(self.best_imprv)

            self.best_rtg = rtg_list[imprv_list.index(self.best_imprv)]
            # print(self.best_rtg)
            # print('output_list', output_list)
            # print('self.best_imprv:', self.best_imprv)
            # top_k = 1  # <=n_jobs
            # max_items = sorted([(val, idx) for idx, val in enumerate(imprv_list)], reverse=True)[:top_k]
            #sorted_imprv = sorted(imprv_list, reverse=True)
            #top_k_rtg = []
            #actions = []
            # for i in range(top_k):
            #     index = max_items[i][1]
            #     top_k_rtg.append(rtg_list[index])
            #     actions = actions_list[index]
                #pass
            #print('Best rtg:', self.best_rtg)
            # best_rtg = top_k_rtg[0]
            # final_imprv_list = Parallel(n_jobs=1)(delayed(rtg2imprv_25)(i) for i in top_k_rtg)
            # print(actions)
            # input()
            #best_imprv = actions2imprv(actions, self.desID, self.init_node, 25)
            '''final_imprv_list = Parallel(n_jobs=4)(delayed(rtg2imprv_25)(i) for i in top_k_rtg)
            print('Best imprv:', final_imprv_list)
            best_imprv = max(final_imprv_list)
            best_rtg = top_k_rtg[final_imprv_list.index(best_imprv)]'''
            #best_imprvs = []athtrk
            #logs['training/train_loss_std'] = np.std(train_losses)
        else:
            current_dir = os.getcwd()
            for file in os.listdir(current_dir):
                if file.endswith('bench') and file.startswith(self.desID):
                    os.remove(os.path.join(current_dir, file))

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            #print(self.eval_fns)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                #print(self.eval_fns)

                print(f'{k}: {v}')

        # max_idx = imprv_list.index(max(imprv_list))
        # print('*'*40)
        # prGreen(self.desID)
        # print(rtg_list[max_idx], end=': ')
        # print('{:.4f}'.format(imprv_list[max_idx]))
        # print('\n')

        print('*'*40)
        prGreen(self.desID)
        print('best_rtg:', self.best_rtg)
        print('best_imprv:', '{:.4f}'.format(self.best_imprv))
        #print('final_imprv:', '{:.4f}'.format(final_imprv))
        print('\n')

        #print(self.collected_datasets[-1])
        #input()
        # finetune_after_training = False
        #
        # if finetune_after_training:
        #     print(self.collected_datasets[-1])
        #     #ft_dataloader = DataLoader(self.collected_datasets, batch_size=1, shuffle=True)
        #     #print(ft_dataloader)
        #     input()
        #     print('Training after generation')
        #     self.model.train()
        #     for _,batch in enumerate(self.collected_datasets):
        #         #batch = batch.to(device)
        #         print(batch)
        #         train_loss = self.train_step(batch)
        #         if self.scheduler is not None:
        #             self.scheduler.step()
        #         print('ft_loss:', train_loss)
        return logs
    def subsequent_mask(size_):
        "Mask out subsequent positions."
        attn_shape = (1, size_, size_)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')  #
        return torch.from_numpy(subsequent_mask) == 0  # 返回上三角为0的torch Tensor

    def train_step(self, batch_data):  #
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=None, target_return=returns,
        )
        print('state_preds', state_preds)
        print('action_preds', action_preds)
        print('reward_preds', reward_preds)

        #note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:,1:], action_target, reward_target[:,1:],
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
