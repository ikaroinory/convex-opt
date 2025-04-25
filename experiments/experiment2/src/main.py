import argparse
import gc
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from datasets import TimeDataset
from evaluate import get_best_performance_data, get_full_err_scores, get_val_performance_data
from models import MGDN
from test import test
from train import train
from utils import build_loc_net, construct_data, get_device, get_fc_graph_struc, get_feature_map, get_graph_struc, get_net_feature_map, set_device


class Main:
    def __init__(self, train_config, env_config):
        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None
        self.net_graph = None

        dataset = self.env_config['dataset']

        phy_train_orig = pd.read_csv('./data/.../train.csv', sep=',', index_col=0)
        phy_test_orig = pd.read_csv('./data/.../test.csv', sep=',', index_col=0)

        net_train_dst = np.genfromtxt('./data/.../train_dst.txt', delimiter=' ', dtype=int)
        net_train_src = np.genfromtxt('./data/.../train_src.txt', delimiter=' ', dtype=int)
        net_train_size = np.genfromtxt('./data/.../train_size.txt', delimiter=' ', dtype=float)
        net_test_dst = np.genfromtxt('./data/.../test_dst.txt', delimiter=' ', dtype=int)
        net_test_src = np.genfromtxt('./data/.../test_src.txt', delimiter=' ', dtype=int)
        net_test_size = np.genfromtxt('./data/.../test_size.txt', delimiter=' ', dtype=int)

        phy_train, phy_test = phy_train_orig, phy_test_orig

        if 'attack' in phy_train.columns:
            phy_train = phy_train.drop(columns=['attack'])

        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_graph_struc(dataset)

        set_device(env_config['device'])
        self.device = get_device()

        phy_fc_edge_index = build_loc_net(fc_struc, list(phy_train.columns), feature_map=feature_map)
        self.phy_graph = torch.tensor(phy_fc_edge_index, dtype=torch.long)

        feature_list, map = get_net_feature_map()
        net_edge_index = get_graph_struc(feature_list, map, feature_map)
        self.net_graph = torch.tensor(net_edge_index, dtype=torch.long)

        self.multi_graph = torch.cat((self.phy_graph, self.net_graph), 1)

        self.feature_map = feature_map

        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        train_dataset_indata = construct_data(phy_train, feature_map, labels=0)
        test_dataset_indata = construct_data(test, feature_map, labels=phy_test.attack.tolist())

        train_dst_dataset = TimeDataset(net_train_dst, self.net_graph, net=None, mode='train', config=cfg)
        train_size_dataset = TimeDataset(net_train_size, self.net_graph, net=None, mode='train', config=cfg)
        train_src_dataset = TimeDataset(net_train_src, self.net_graph, net=None, mode='train', config=cfg)
        test_dst_dataset = TimeDataset(net_test_dst, self.net_graph, net=None, mode='test', config=cfg)
        test_size_dataset = TimeDataset(net_test_size, self.net_graph, net=None, mode='test', config=cfg)
        test_src_dataset = TimeDataset(net_test_src, self.net_graph, net=None, mode='test', config=cfg)

        train_net_y = torch.cat((train_src_dataset.y, train_size_dataset.y, train_dst_dataset.y), 1).to(self.device)
        net_test_y = torch.cat((test_src_dataset.y, test_size_dataset.y, test_dst_dataset.y), 1).to(self.device)
        net_test_x = torch.cat((test_src_dataset.x, test_size_dataset.x, test_dst_dataset.x), 1).to(self.device)

        train_dataset = TimeDataset(train_dataset_indata, self.phy_graph, net=train_net_y, mode='train', config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, self.phy_graph, net=net_test_y, mode='test', config=cfg)
        bnum = torch.tensor(net_test_x).shape[0]

        net_test_x = net_test_x.reshape(-1, 51, 45)
        net_test_x = torch.tensor(net_test_x)
        net_test_x = net_test_x.reshape(bnum, test_dataset.x.shape[1], -1).to(self.device)
        net_test_x = net_test_x[:test_dataset.x.size()[0], :, :]
        test_dataset.x = test_dataset.x.to(self.device)
        n_node = test_dataset.py.size()[0]
        test_dataset.ny = test_dataset.ny[:n_node, :].to(self.device)
        test_dataset.labels = test_dataset.labels[:n_node].to(self.device)
        test_dataset.x = torch.cat((test_dataset.x, net_test_x), 2)
        train_dataset.x = train_dataset.x.to(self.device)

        train_dataset.x = train_dataset.x[:train_src_dataset.x.size()[0], :, :].to(self.device)
        train_src_dataset.x = train_src_dataset.x.to(self.device)
        train_dst_dataset.x = train_dst_dataset.x.to(self.device)
        train_size_dataset.x = train_size_dataset.x.to(self.device)

        train_dataset.x = torch.cat((train_dataset.x, train_src_dataset.x), 2).to(self.device)
        train_dataset.x = torch.cat((train_dataset.x, train_size_dataset.x), 2).to(self.device)
        train_dataset.x = torch.cat((train_dataset.x, train_dst_dataset.x), 2).to(self.device)

        gc.collect()
        torch.cuda.empty_cache()

        train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['batch'], val_ratio=train_config['val_ratio'])

        test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'], shuffle=False, num_workers=0)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        edge_index_sets = [self.multi_graph]

        self.val_dataloader = val_dataloader

        self.model = MGDN(
            edge_index_sets,
            len(feature_map),
            dim=train_config['dim'],
            out_layer_num=train_config['out_layer_num'],
            out_layer_inter_dim=train_config['out_layer_inter_dim'],
            topk=train_config['topk']
        ).to(self.device)

    @staticmethod
    def get_loaders(train_dataset, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index + val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index + val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)

        train_dataloader = DataLoader(train_subset, batch_size=batch, shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch, shuffle=False)

        return train_dataloader, val_dataloader

    @staticmethod
    def get_expand(n, dataset):
        q = dataset.size()[0]
        while q < n:
            dataset = torch.cat((dataset, dataset), 0)
            q = dataset.size()[0]
        if q == n:
            return dataset
        else:
            dataset = dataset[:n]
            return dataset

    def get_score(self, test_result, val_result):

        np_test_result = np.array(test_result, dtype=object)

        test_labels = np_test_result[2, :, 0].tolist()

        test_scores, normal_scores = get_full_err_scores(test_result, val_result)

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1)
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)

        print('=========================** Result **============================\n')

        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info
        elif self.env_config['report'] == 'val':
            info = top1_val_info

        print('F1 score: {info[0]}')
        print(info[0])
        print('precision: {info[1]}')
        print(info[1])
        print('recall: {info[2]}')
        print(info[2])

    def get_save_path(self):

        dir_path = self.env_config['save_path']

        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr

        paths = [
            './pretrained/' + dir_path + '/best_' + datestr + '.pt',
            './results/' + dir_path + '/' + datestr + '.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths

    def run(self):
        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
        else:
            model_save_path = self.get_save_path()[0]

            self.train_log, nlgarph = train(
                mul_adj=self.multi_graph,
                phy_adj=self.phy_graph,
                net_adj=self.net_graph,
                model=self.model,
                save_path=model_save_path,
                config=train_config,
                train_dataloader=self.train_dataloader
            )

        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)

        avg_loss, self.phy_test_result, self.net_test_result = test(
            best_model,
            self.test_dataloader,
            phy_adj=self.phy_graph,
            net_adj=self.net_graph,
            mul_adj=self.multi_graph
        )
        avg_loss_val, self.phy_val_result, self.net_val_result = test(
            best_model,
            self.val_dataloader,
            phy_adj=self.phy_graph,
            net_adj=self.net_graph,
            mul_adj=self.multi_graph
        )

        self.get_score(self.phy_test_result, self.phy_val_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type=int, default=32)
    parser.add_argument('-epoch', help='train epoch', type=int, default=100)
    parser.add_argument('-slide_win', help='slide_win', type=int, default=15)
    parser.add_argument('-dim', help='dimension', type=int, default=64)
    parser.add_argument('-slide_stride', help='slide_stride', type=int, default=5)
    parser.add_argument('-save_path_pattern', help='save path pattern', type=str, default='')
    parser.add_argument('-dataset', help='wadi / swat', type=str, default='swat')
    parser.add_argument('-device', help='cuda / cpu', type=str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type=int, default=0)
    parser.add_argument('-comment', help='experiment comment', type=str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type=int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type=int, default=256)
    parser.add_argument('-decay', help='decay', type=float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type=float, default=0.1)
    parser.add_argument('-topk', help='topk num', type=int, default=20)
    parser.add_argument('-report', help='best / val', type=str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type=str, default='./pretrained/best_05|24-10:24:32.pt')

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
    }

    env_config = {
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path
    }

    main = Main(train_config, env_config)
    main.run()
