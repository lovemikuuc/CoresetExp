#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/4 9:16
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : benchmark.py
# @Description :

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from apricot import FacilityLocationSelection

import train
from config import load_dataset_label_names
from models import fetch_classifier
from plot import plot_matrix
from statistic import stat_acc_f1, stat_results
from utils import get_device, handle_argv, IMUDataset, load_classifier_data_config, \
    FFTDataset, prepare_classifier_dataset, Preprocess4Normalization

import os
import random
import time

validate = None
test = None
new_data = None
new_label = None
core_factor = 0.05

# Q: are above  global variables??
# 



def classify_benchmark(args, label_index, training_rate, label_rate, balance=True, method=None):
    
    data, labels, train_cfg, model_cfg, dataset_cfg = load_classifier_data_config(args)
    label_names, label_num = load_dataset_label_names(dataset_cfg, label_index)
    data_train, label_train, data_vali, label_vali, data_test, label_test \
        = prepare_classifier_dataset(data, labels, label_index=label_index, training_rate=training_rate
                                    , label_rate=label_rate, merge=model_cfg.seq_len
                                    , seed=train_cfg.seed, balance=balance)
    pipeline = [Preprocess4Normalization(model_cfg.input)]
    
    
    global new_data
    global new_label
    new_data = data_train
    new_label = label_train
    
    #new_data = np.concatenate((data_train, data_vali, data_test), axis=0)
    #new_label = np.concatenate((label_train, label_vali, label_test), axis=0)
    print(new_data.shape)
    print(data_test.shape)
    print(data_vali.shape)

    
    
    global validate
    global test
    
    if method != 'deepsense':
        data_set_train = IMUDataset(data_train, label_train, pipeline=pipeline)
        test = IMUDataset(data_test, label_test, pipeline=pipeline)
        validate = IMUDataset(data_vali, label_vali, pipeline=pipeline)
    else:
        data_set_train = FFTDataset(data_train, label_train, pipeline=pipeline)
        test = FFTDataset(data_test, label_test, pipeline=pipeline)
        validate = FFTDataset(data_vali, label_vali, pipeline=pipeline)
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_test = DataLoader(test, shuffle=False, batch_size=train_cfg.batch_size)
    data_loader_vali = DataLoader(validate, shuffle=False, batch_size=train_cfg.batch_size)

    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss()
    model = fetch_classifier(method, model_cfg, input=model_cfg.input, output=label_num)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)  # , weight_decay=0.95
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, get_device(args.gpu))

    def func_loss(model, batch):
        inputs, label = batch
        logits = model(inputs, True)
        loss = criterion(logits, label)
        return loss

    def func_forward(model, batch):
        inputs, label = batch
        logits = model(inputs, False)
        return logits, label

    def func_evaluate(label, predicts):
        stat = stat_acc_f1(label.cpu().numpy(), predicts.cpu().numpy())
        return stat

    trainer.train(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, data_loader_vali)
    label_estimate_test = trainer.run(func_forward, None, data_loader_test)
    return label_test, label_estimate_test



def classify_benchmark2(args, label_index, training_rate, label_rate, balance=True, method=None):
    
    data, labels, train_cfg, model_cfg, dataset_cfg = load_classifier_data_config(args)
    label_names, label_num = load_dataset_label_names(dataset_cfg, label_index)
    data_train, label_train, data_vali, label_vali, data_test, label_test \
        = prepare_classifier_dataset(data, labels, label_index=label_index, training_rate=training_rate
                                    , label_rate=label_rate, merge=model_cfg.seq_len
                                    , seed=train_cfg.seed, balance=balance)
    pipeline = [Preprocess4Normalization(model_cfg.input)]
    
    global new_data
    global new_label
    new_data = data_train
    new_label = label_train
    global validate
    global test




    # core_num = int(data.shape[0] * core_factor)
    # data_dim2 =  data.reshape(data.shape[0], data.shape[1] * data.shape[2])

    core_num = int(new_data.shape[0] * core_factor)
    data_dim2 =  new_data.reshape(new_data.shape[0], new_data.shape[1] * new_data.shape[2])

    idx = apricot_coreset(data_dim2, core_num)

    # data_coreset = data[idx]
    # label_coreset = label[idx]

    data_coreset = new_data[idx]
    label_coreset = new_label[idx]
    print(data_coreset.shape)
    print(data_test.shape)
    print(data_vali.shape)

    
    if method != 'deepsense':
        data_set_train = IMUDataset(data_coreset, label_coreset, pipeline=pipeline)
        test = IMUDataset(data_test, label_test, pipeline=pipeline)
        validate = IMUDataset(data_vali, label_vali, pipeline=pipeline)
       
    else:
        data_set_train = FFTDataset(data_coreset, label_coreset, pipeline=pipeline)
        test = FFTDataset(data_test, label_test, pipeline=pipeline)
        validate = FFTDataset(data_vali, label_vali, pipeline=pipeline)
     


    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_test = DataLoader(test, shuffle=False, batch_size=train_cfg.batch_size)
    data_loader_vali = DataLoader(validate, shuffle=False, batch_size=train_cfg.batch_size)

    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss()
    model = fetch_classifier(method, model_cfg, input=model_cfg.input, output=label_num)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)  # , weight_decay=0.95
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, get_device(args.gpu))

    def func_loss(model, batch):
        inputs, label = batch
        logits = model(inputs, True)
        loss = criterion(logits, label)
        return loss

    def func_forward(model, batch):
        inputs, label = batch
        logits = model(inputs, False)
        return logits, label

    def func_evaluate(label, predicts):
        stat = stat_acc_f1(label.cpu().numpy(), predicts.cpu().numpy())
        return stat

    trainer.train(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, data_loader_vali)
    label_estimate_test = trainer.run(func_forward, None, data_loader_test)
    return label_test, label_estimate_test




def classify_benchmark3(args, label_index, training_rate, label_rate, balance=True, method=None):
    
    data, labels, train_cfg, model_cfg, dataset_cfg = load_classifier_data_config(args)
    label_names, label_num = load_dataset_label_names(dataset_cfg, label_index)
    data_train, label_train, data_vali, label_vali, data_test, label_test \
        = prepare_classifier_dataset(data, labels, label_index=label_index, training_rate=training_rate
                                    , label_rate=label_rate, merge=model_cfg.seq_len
                                    , seed=train_cfg.seed, balance=balance)
    pipeline = [Preprocess4Normalization(model_cfg.input)]
    
    t = 1000 * time.time()
    set_seeds(int(t) % 2**32)


    # core_num = int(data.shape[0] * core_factor)
    # data_dim2 =  data.reshape(data.shape[0], data.shape[1] * data.shape[2])

    core_num = int(new_data.shape[0] * core_factor)
    
    print(core_num)
    idx_random = random.sample(range(0,new_data.shape[0]),core_num)


    data_random = new_data[idx_random]
    label_random = new_label[idx_random]
    print(data_random.shape)
    print(data_test.shape)
    print(data_vali.shape)

    
    if method != 'deepsense':
        data_set_train = IMUDataset(data_random, label_random, pipeline=pipeline)
    else:
        data_set_train = FFTDataset(data_random, label_random, pipeline=pipeline)

    
    
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_test = DataLoader(test, shuffle=False, batch_size=train_cfg.batch_size)
    data_loader_vali = DataLoader(validate, shuffle=False, batch_size=train_cfg.batch_size)

    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss()
    model = fetch_classifier(method, model_cfg, input=model_cfg.input, output=label_num)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)  # , weight_decay=0.95
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, get_device(args.gpu))

    def func_loss(model, batch):
        inputs, label = batch
        logits = model(inputs, True)
        loss = criterion(logits, label)
        return loss

    def func_forward(model, batch):
        inputs, label = batch
        logits = model(inputs, False)
        return logits, label

    def func_evaluate(label, predicts):
        stat = stat_acc_f1(label.cpu().numpy(), predicts.cpu().numpy())
        return stat

    trainer.train(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, data_loader_vali)
    label_estimate_test = trainer.run(func_forward, None, data_loader_test)
    return label_test, label_estimate_test




def apricot_coreset(X, n):
    selector = FacilityLocationSelection(n,optimizer='naive',verbose=True)
    selector.fit(X)
    return selector.ranking

def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    train_rate = 0.8
    balance = True
    label_rate = 0.05
    method = "gru"

    args = handle_argv('bench_' + method, 'train.json', method)

    # is the following args user input?
    # yes, it is user input
    #print("Running classify_benchmark1")
    #label_test, label_estimate_test = classify_benchmark(args, args.label_index, train_rate, label_rate, balance=balance, method=method)

    #label_names, label_num = load_dataset_label_names(args.dataset_cfg, args.label_index)
    #cc, matrix, f1 = stat_results(label_test, label_estimate_test)
    # matrix_norm = plot_matrix(matrix, label_names)


    # repeat above process for classify_benchmark2
    print("Running classify_benchmark2")
    label_test2, label_estimate_test2 = classify_benchmark2(args, args.label_index, train_rate, label_rate, balance=balance, method=method)

    label_names2, label_num2 = load_dataset_label_names(args.dataset_cfg, args.label_index)
    acc2, matrix2, f12 = stat_results(label_test2, label_estimate_test2)
    #matrix_norm2 = plot_matrix(matrix2, label_names2)


    # repeat above process for classify_benchmark3
    for i in range(8):
        set_seeds(i*50+i*70-i)
        print("Running classify_benchmark3 Attempt {}".format(i))
        label_test3, label_estimate_test3 = classify_benchmark3(args, args.label_index, train_rate, label_rate, balance=balance, method=method)
    

    label_names3, label_num3 = load_dataset_label_names(args.dataset_cfg, args.label_index)
    acc3, matrix3, f13 = stat_results(label_test3, label_estimate_test3)
    # matrix_norm3 = plot_matrix(matrix3, label_names3)
    






