#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.8
import os

import pickle
from Hetero_Client import Hetero_Client, evaluate
from agg.avg import *
from Text import DatasetLM
from utils.options import args_parser
from util import get_dataset_mnist_extr_noniid, get_dataset_cifar10_extr_noniid, get_dataset_cifar100_extr_noniid, \
    get_dataset_tiny_extr_noniid, train_val_test_image
from Models import all_models
from data.reddit.user_data import data_process
import warnings

warnings.filterwarnings('ignore')

args = args_parser()

if __name__ == "__main__":
    config = args
    config.mask = True
    config.learnable = True
    config.personalized = True

    torch.cuda.set_device(config.gpu)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if config.gpu != -1:
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')
    config.device = device

    dataset_train, dataset_val, dataset_test, data_num = {}, {}, {}, {}
    if config.dataset == 'mnist':
        idx_vals = []
        total_train_data, total_test_data, dataset_train_idx, dataset_test_idx = get_dataset_mnist_extr_noniid(
            config.nusers,
            config.nclass,
            config.nsamples,
            config.rate_unbalance)
        for i in range(config.nusers):
            idx_val, dataset_train[i], dataset_val[i], dataset_test[i] = train_val_test_image(total_train_data,
                                                                                              list(
                                                                                                  dataset_train_idx[i]),
                                                                                              total_test_data,
                                                                                              list(dataset_test_idx[i]))
            idx_vals.append(idx_val)
            data_num[i] = len(dataset_train[i])
    elif config.dataset == 'cifar10':
        idx_vals = []
        total_train_data, total_test_data, dataset_train_idx, dataset_test_idx = get_dataset_cifar10_extr_noniid(
            config.nusers,
            config.nclass,
            config.nsamples,
            config.rate_unbalance)
        for i in range(config.nusers):
            idx_val, dataset_train[i], dataset_val[i], dataset_test[i] = train_val_test_image(total_train_data,
                                                                                              list(
                                                                                                  dataset_train_idx[i]),
                                                                                              total_test_data,
                                                                                              list(dataset_test_idx[i]))
            idx_vals.append(idx_val)
            data_num[i] = len(dataset_train[i])
    elif args.dataset == 'cifar100':
        idx_vals = []
        total_train_data, total_test_data, dataset_train_idx, dataset_test_idx = get_dataset_cifar100_extr_noniid(
            args.nusers,
            args.nclass,
            args.nsamples,
            args.rate_unbalance)
        for i in range(args.nusers):
            idx_val, dataset_train[i], dataset_val[i], dataset_test[i] = train_val_test_image(total_train_data,
                                                                                              list(
                                                                                                  dataset_train_idx[i]),
                                                                                              total_test_data,
                                                                                              list(dataset_test_idx[i]))
            idx_vals.append(idx_val)
            data_num[i] = len(dataset_train[i])
    elif config.dataset == 'reddit':
        # dataload
        data_dir = 'data/reddit/train/'
        with open('data/reddit/reddit_vocab.pck', 'rb') as f:
            vocab = pickle.load(f)
        nvocab = vocab['size']
        config.nvocab = nvocab
        train_data, val_data, test_data = data_process(data_dir, nvocab, config.nusers)
        for i in range(config.nusers):
            dataset_train[i] = DatasetLM(train_data[i], vocab['vocab'])
            dataset_val[i] = DatasetLM(val_data[i], vocab['vocab'])
            dataset_test[i] = DatasetLM(test_data[i], vocab['vocab'])
            data_num[i] = len(train_data[i])
    elif args.dataset == 'tinyimagenet':
        idx_vals = []
        total_train_data, total_test_data, dataset_train_idx, dataset_test_idx = get_dataset_tiny_extr_noniid(
            args.nusers,
            args.nclass,
            args.nsamples,
            args.rate_unbalance)
        for i in range(args.nusers):
            idx_val, dataset_train[i], dataset_val[i], dataset_test[i] = train_val_test_image(total_train_data,
                                                                                              list(
                                                                                                  dataset_train_idx[i]),
                                                                                              total_test_data,
                                                                                              list(dataset_test_idx[i]))
            idx_vals.append(idx_val)
            data_num[i] = len(dataset_train[i])

    best_val_acc = None
    os.makedirs(f'./log/{config.dataset}/', exist_ok=True)
    model_saved = './log/{}/model_FLST_{}.pt'.format(config.dataset, config.seed)

    # 确定哪些层可以被mask
    config.mask_weight_indicator = []
    config.personal_layers = []
    model_indicator = all_models[config.dataset](config, device)
    model_weight = copy.deepcopy(model_indicator.state_dict())
    layers = list(model_weight.keys())
    layers_name = []
    for key in layers:
        if 'weight' in key:
            layers_name.append(key)
    first_layer = layers_name[0]
    last_layer = layers_name[-1]
    model_indicator.to(device)
    if config.mask:
        model_indicator.label_mask_weight()
        mask_weight_indicator = model_indicator.mask_weight_indicator
        # if first_layer in mask_weight_indicator:
        #     mask_weight_indicator = mask_weight_indicator[1:]
        if last_layer in mask_weight_indicator:
            mask_weight_indicator = mask_weight_indicator[:-1]
        config.mask_weight_indicator = copy.deepcopy(mask_weight_indicator)
    if config.personalized:
        model_indicator = all_models[config.dataset](config, device)
        personal_layers = []
        personal_layers.append(layers[-2])
        personal_layers.append(layers[-1])
        config.personal_layers = personal_layers

    # initialized global model
    net_glob = all_models[config.dataset](config, device)
    net_glob = net_glob.to(device)
    w_glob = net_glob.state_dict()
    # initialize clients
    clients = {}
    client_ids = []
    for client_id in range(config.nusers):
        cl = Hetero_Client(config, device, client_id, dataset_train[client_id], dataset_val[client_id],
                           dataset_test[client_id],
                           local_net=all_models[config.dataset](config, device), mask_rate=config.mask_rate)
        clients[client_id] = cl
        client_ids.append(client_id)
        torch.cuda.empty_cache()

    for round in range(config.rounds):
        upload_cost_round, uptime = [], []
        download_cost_round, down_time = [], []
        compute_flops_round, training_time = [], []

        net_glob.train()
        w_locals, loss_locals, acc_locals = [], [], []
        m = max(int(config.frac * len(clients)), 1)
        idxs_users = np.random.choice(len(clients), m, replace=False)  # 随机采样client
        total_num = 0
        for idx in idxs_users:
            client = clients[idx]
            i = client_ids.index(idx)
            train_result = client.update_weights_leanable(w_server=copy.deepcopy(w_glob), round=round)
            w_locals.append(train_result['state'])

            download_cost_round.append(train_result['dl_cost'])
            down_time.append(train_result['dl_cost'] / 8 / 1024 / 1024 / 110.6)
            upload_cost_round.append(train_result['ul_cost'])
            uptime.append(train_result['ul_cost'] / 8 / 1024 / 1024 / 14.0)
            compute_flops_round.append(train_result['train_flops'])

        w_glob = avg(w_locals, w_glob, config, device)
        if config.dataset == 'reddit':
            w_glob[last_layer] = copy.deepcopy(w_glob[first_layer])
        net_glob.load_state_dict(w_glob)
        net_glob = net_glob.to(device)

        train_loss, train_acc = [], []
        val_loss, val_acc = [], []
        per_test_loss, per_test_acc = [], []
        glob_test_loss, glob_test_acc = [], []
        if config.personalized:
            for _, c in clients.items():
                local_model = copy.deepcopy(net_glob)
                local_params = copy.deepcopy(w_glob)
                if c.final_parameters is not None:
                    if config.dataset == 'reddit':
                        for key in config.mask_weight_indicator:
                            if key in c.final_parameters and 'encoder' not in key:
                                local_params[key] = copy.deepcopy(c.final_parameters[key])
                        local_model.load_state_dict(copy.deepcopy(local_params))
                    else:
                        local_model.load_state_dict(copy.deepcopy(c.final_parameters))
                else:
                    for key in args.personal_layers:
                        local_params[key] = c.local_net.state_dict()[key]
                    local_model.load_state_dict(copy.deepcopy(local_params))
                train_res = evaluate(config, c.traindata_loader, local_model, device)
                train_loss.append(train_res[0])
                train_acc.append(train_res[1])
                per_val_res = evaluate(config, c.valdata_loader, local_model, device)
                val_loss.append(per_val_res[0])
                val_acc.append(per_val_res[1])
                per_test_res = evaluate(config, c.testdata_loader, local_model, device)
                per_test_loss.append(per_test_res[0])
                per_test_acc.append(per_test_res[1])
            train_loss, train_acc = np.array(train_loss)[~np.isnan(np.array(train_loss))], np.array(train_acc)[
                ~np.isnan(np.array(train_acc))]
            val_loss, val_acc = np.array(val_loss)[~np.isnan(np.array(val_loss))], np.array(val_acc)[
                ~np.isnan(np.array(val_acc))]
            per_test_loss, per_test_acc = np.array(per_test_loss)[~np.isnan(np.array(per_test_loss))], \
                np.array(per_test_acc)[~np.isnan(np.array(per_test_acc))]
            print('\nRound {}, Train loss: {:.5f}, train accuracy: {:.5f}'.format(round, np.mean(train_loss),
                                                                                  np.mean(train_acc)))
            print('Max upload cost: {:.3f} MB, Max download cost: {:.3f} '
                  'MB'.format(max(upload_cost_round) / 8 / 1024 / 1024, max(download_cost_round) / 8 / 1024 / 1024))
            print('Sum compute flops cost: {:.3f} MB'.format(sum(compute_flops_round) / 1e6))
            print("Validation loss: {:.5f}, val accuracy: {:.5f}".format(np.mean(val_loss),
                                                                         np.mean(val_acc)), flush=True)
            print("test loss: {:.5f}, test accuracy: {:.5f}".format(np.mean(per_test_loss),
                                                                    np.mean(per_test_acc)), flush=True)
        else:
            for _, c in clients.items():
                train_res = evaluate(config, c.traindata_loader, net_glob, device)
                train_loss.append(train_res[0])
                train_acc.append(train_res[1])
                glob_val_res = evaluate(config, c.valdata_loader, net_glob, device)
                val_loss.append(glob_val_res[0])
                val_acc.append(glob_val_res[1])
                glob_test_res = evaluate(config, c.testdata_loader, copy.deepcopy(net_glob), device)
                glob_test_loss.append(glob_test_res[0])
                glob_test_acc.append(glob_test_res[1])
            glob_test_loss, glob_test_acc = np.array(glob_test_loss)[~np.isnan(np.array(glob_test_loss))], \
                np.array(glob_test_acc)[~np.isnan(np.array(glob_test_acc))]
            train_loss, train_acc = np.array(train_loss)[~np.isnan(np.array(train_loss))], np.array(train_acc)[
                ~np.isnan(np.array(train_acc))]
            val_loss, val_acc = np.array(val_loss)[~np.isnan(np.array(val_loss))], np.array(val_acc)[
                ~np.isnan(np.array(val_acc))]
            print('\nRound {}, Train loss: {:.5f}, train accuracy: {:.5f}'.format(round, np.mean(train_loss),
                                                                                  np.mean(train_acc)))
            print('Max upload cost: {:.3f} MB, Max download cost: {:.3f} '
                  'MB'.format(max(upload_cost_round) / 8 / 1024 / 1024, max(download_cost_round) / 8 / 1024 / 1024))
            print('Sum compute flops cost: {:.3f} MB'.format(sum(compute_flops_round) / 1e6))
            print("Validation loss: {:.5f}, val accuracy: {:.5f}".format(np.mean(val_loss), np.mean(val_acc)))
            print("test loss: {:.5f}, test accuracy: {:.5f}".format(np.mean(glob_test_loss), np.mean(glob_test_acc)))