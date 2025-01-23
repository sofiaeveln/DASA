
import argparse
import os
from pathlib import Path
import socket
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Semiformer import DASA, DASAConfig
import utils
import logging
import sys
from datetime import datetime

SPEC_DICT = {
    'sharemlp': ['value_mlp', 'value_transform']
}

CITY_DICT = {
    'chengdu': 'Chengdu',
    'shenzhen': 'Shenzhen',
    'pems-bay': 'PemsBay',
    'metr-la': 'MetrLA'
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--cuda', default="0", type=str)
    parser.add_argument('--dtype', default='float32', type=str)
    parser.add_argument('--method', default='DASA', type=str)
    parser.add_argument('--spec_type', default='sharemlp', type=str)
    parser.add_argument('--domain_specific_params', nargs='+', default=['value_mlp', 'value_transform'])
    parser.add_argument('--data', type=str, default='chengdu')  
    parser.add_argument('--load_type', default='best', choices=['best', 'last'])
    parser.add_argument('--train_cities', nargs='+', default=['metr-la', 'shenzhen', 'pems-bay'])
    parser.add_argument('--out_dir', default="out", type=str)
    parser.add_argument('--init_from', default="resume", type=str)
    parser.add_argument('--seq_len', default=12, type=int)
    parser.add_argument('--horizons', nargs='+', type=int, default=[12], help='List of horizons to predict')

    parser.add_argument('--input_dim', default=2, type=int, help='输入基础特征维度')
    parser.add_argument('--tod_embedding_dim', default=12, type=int, help='tod嵌入维度，0则不启用tod嵌入')
    parser.add_argument('--dow_embedding_dim', default=6, type=int, help='dow嵌入维度，0则不启用dow嵌入')
    parser.add_argument('--spatial_embedding_dim', default=6, type=int, help='空间嵌入维度，0则不启用')
    parser.add_argument('--adaptive_embedding_dim', default=36, type=int, help='自适应嵌入维度，0则不启用')
    parser.add_argument('--steps_per_day', default=288, type=int, help='一天的时间步数，用于tod/dow计算')
    parser.add_argument('--output_dim', default=1, type=int, help='预测输出特征维度(通常为1)')
    
    parser.add_argument('--blocksize', default=8, type=int)
    parser.add_argument('--blocknum', default=4, type=int)
    parser.add_argument('--factors', default=1, type=int)


    args = parser.parse_args()

    utils.set_random_seed(args.seed)
    args.hostname = socket.gethostname()
    args.datapath = '/root/data'
    args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else torch.device("cpu"))
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}.get(args.dtype, torch.float32)
    args.ctx = torch.autocast(device_type='cuda', dtype=ptdtype) if 'cuda' in str(args.device) else None

    args.out_dir = os.path.join(args.out_dir, 'main')
    # postfix = f'me{5}-ce{2}-te{50}-mlr{1e-3}-tlr{0.01}'
    postfix = f'me{args.meta_epochs}-ce{args.city_epochs}-te{args.test_epochs}-mlr{args.meta_lr}-tlr{args.update_lr}'
    train_str = f'{args.seed}-{args.spec_type}-ln{1}-{postfix}'
    post_str = '-post'
    test_str = f'{args.seed}-{args.spec_type}{post_str}-{args.load_type}-ln{1}-{postfix}'

    args.out_dir = os.path.join(args.out_dir, f'ln{1}')
    path_test_model = os.path.join(args.out_dir, f'{args.data}_{train_str}_ckpt.pth')
    path_test_model_last = os.path.join(args.out_dir, f'{args.data}_{train_str}_ckpt_last.pth')
    path_test_res = os.path.join(args.out_dir, f'{args.data}_{test_str}_res.npz')

    log_dir = './logs'
    log_prefix = f'{args.method}-{args.data}-{train_str}-sample-{args.hostname}-gpu{args.cuda}'
    logger = utils.set_logger(log_dir=log_dir, log_prefix=log_prefix)
    logger.info(args)

    if args.init_from == 'resume':
        if args.load_type == 'best':
            logger.info(f'Load test model from {path_test_model}')
            ckpt = torch.load(path_test_model, map_location=args.device)
        else:
            logger.info(f'Load test model from {path_test_model_last}')
            ckpt = torch.load(path_test_model_last, map_location=args.device)
        colaconf = COLAConfig(**ckpt['model_args'])

        colaconf.input_dim = args.input_dim
        colaconf.tod_embedding_dim = args.tod_embedding_dim
        colaconf.dow_embedding_dim = args.dow_embedding_dim
        colaconf.spatial_embedding_dim = args.spatial_embedding_dim
        colaconf.adaptive_embedding_dim = args.adaptive_embedding_dim
        colaconf.steps_per_day = args.steps_per_day
        colaconf.output_dim = args.output_dim
        colaconf.blocksize = args.blocksize
        colaconf.blocknum = args.blocknum
        colaconf.factors = args.factors
        
        model = COLA(colaconf)
        state_dict = ckpt['model']
        model.load_state_dict(state_dict)
    else:
        logger.error("Initialization type is incorrect! Use --init_from 'resume'")
        exit(1)

    model.eval()
    model.to(args.device)

    # 使用utils读取数据集
    data_neural, _ = utils.read_meta_datasets(
        train_cities=args.train_cities,
        test_city=args.data,
        path=args.datapath,
        tod_embedding_dim=args.tod_embedding_dim,
        dow_embedding_dim=args.dow_embedding_dim,
        steps_per_day=args.steps_per_day
    )

    # 构建测试数据
    dataset = data_neural[args.data]['dataset'][:3 * args.steps_per_day]  # 假设测试使用前3天数据
    splits = utils.generate_data(dataset, args.seq_len, args.horizons)

    x_test = splits['test']['x']
    y_tests = splits['test']['y']

    test_loader = [(torch.FloatTensor(x_test), {h: torch.FloatTensor(y_tests[h]) for h in args.horizons})]

    test_loader = [(x.to(args.device), {h: y_tests[h].to(args.device) for h in args.horizons})
                    for x, y_tests in test_loader]
    
    with torch.no_grad():
        if args.ctx is not None:
            with args.ctx:
                results = model.evaluate(test_loader)
        else:
            results = model.evaluate(test_loader)

        for h in args.horizons:
            mae = results[h]['MAE']
            mape = results[h]['MAPE']
            mse = results[h]['MSE']
            rmse = results[h]['RMSE']
            logger.info(f"Horizon {h}: MAE={mae:.5f}, MAPE={mape:.3f}%, MSE={mse:.5f}, RMSE={rmse:.5f}")

        np.savez(path_test_res, **{f'h{h}_mae': results[h]['MAE'] for h in args.horizons},
                              **{f'h{h}_mape': results[h]['MAPE'] for h in args.horizons},
                              **{f'h{h}_mse': results[h]['MSE'] for h in args.horizons},
                              **{f'h{h}_rmse': results[h]['RMSE'] for h in args.horizons})



