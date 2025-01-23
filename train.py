import argparse
import os
from pathlib import Path
import socket
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import utils
from Semiformer import DASA, DASAConfig
import sys
from datetime import datetime
import logging
import math

# 设置环境变量
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

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
    # 基本参数
    parser.add_argument('--seed', default=0, type=int, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--cuda', default="0", type=str)
    parser.add_argument('--dtype', default='float32', type=str)
    parser.add_argument('--method', default='DASA', type=str)
    parser.add_argument('--train_cities', nargs='+', default=['chengdu', 'metr-la', 'shenzhen'])
    parser.add_argument('--data', type=str, default='pems-bay')
    parser.add_argument('--spec_type', default='sharemlp', type=str)
    parser.add_argument('--domain_specific_params', nargs='+', default=['value_mlp', 'value_transform'])
    parser.add_argument('--datapath', default='/root/data', type=str)
    parser.add_argument('--out_dir', default="out", type=str)
    parser.add_argument('--seq_len', default=12, type=int)
    parser.add_argument('--horizons', nargs='+', type=int, default=[3, 6, 12, 24], help='List of horizons to predict')
    parser.add_argument('--load_type', default='best', choices=['best', 'last'], type=str,
                        help="Type of model checkpoint to load (e.g., 'best' or 'last').")
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=5e-5)
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-3)
    parser.add_argument('--meta_epochs', default=5, type=int)
    parser.add_argument('--city_epochs', default=1, type=int)
    parser.add_argument('--test_epochs', default=50, type=int)
    
    # 模型参数
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--n_head', default=4, type=int)
    parser.add_argument('--n_layer', default=6, type=int) 
    parser.add_argument('--n_linear', default=1, type=int)
    parser.add_argument('--n_embd', default=64, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--bias', default=False, type=bool)

    # 优化参数
    parser.add_argument('--grad_clip', default=1.0, type=float, help='clip gradients at this value, or disable if == 0.0')
    parser.add_argument('--eval_only', default=False, type=bool)
    parser.add_argument('--eval_interval', default=2, type=int)

     # 新增embedding相关参数
    parser.add_argument('--input_dim', default=2, type=int)
    parser.add_argument('--tod_embedding_dim', default=8, type=int)
    parser.add_argument('--dow_embedding_dim', default=4, type=int)
    parser.add_argument('--spatial_embedding_dim', default=24, type=int)
    parser.add_argument('--adaptive_embedding_dim', default=8, type=int)
    parser.add_argument('--steps_per_day', default=288, type=int)
    parser.add_argument('--output_dim', default=1, type=int)
    
    parser.add_argument('--temporal_layers', default=1, type=int, help='Temporal Transformer层数')
    parser.add_argument('--spatial_layers', default=1, type=int, help='Spatial Transformer层数')
    
    parser.add_argument('--blocksize', default=8, type=int)
    parser.add_argument('--blocknum', default=30, type=int)
    parser.add_argument('--factors', default=1, type=int)

    args = parser.parse_args()
    # 根据spec_type从SPEC_DICT获取最终的domain_specific_params
    args.domain_specific_params = SPEC_DICT[args.spec_type]

    # 设置随机种子和设备
    utils.set_random_seed(args.seed)
    args.hostname = socket.gethostname()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else torch.device("cpu"))
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}.get(args.dtype, torch.float32)
    args.ctx = torch.autocast(device_type='cuda', dtype=ptdtype) if 'cuda' in str(device) else None
    args.out_dir = os.path.join(args.out_dir, 'main')

    # 设置日志记录
    postfix = f'me{args.meta_epochs}-ce{args.city_epochs}-te{args.test_epochs}-mlr{args.meta_lr}-tlr{args.update_lr}'
    train_str = f'{args.seed}-{args.spec_type}-ln{args.n_linear}-{postfix}'
    post_str = '-post'
    test_str = f'{args.seed}-{args.spec_type}{post_str}-{args.load_type}-ln{args.n_linear}-{postfix}'

    args.out_dir = os.path.join(args.out_dir, f'ln{args.n_linear}')
    os.makedirs(args.out_dir, exist_ok=True)
    path_meta_model = f'{args.out_dir}/{args.data}_{train_str}_meta.pth'
    path_test_model = f'{args.out_dir}/{args.data}_{train_str}_ckpt.pth'
    path_test_model_last = f'{args.out_dir}/{args.data}_{train_str}_ckpt_last.pth'
    path_test_res = os.path.join(args.out_dir, f'{args.data}_{test_str}_res.npz')

    log_dir = './logs'
    log_prefix = f'{args.method}-{args.data}-{train_str}-train-{args.hostname}-gpu{args.cuda}'
    logger = utils.set_logger(log_dir=log_dir, log_prefix=log_prefix)
    logger.info(args)

    # 根据参数读入多源数据集，并添加tod/dow特征
    data_neural, meta_num_nodes = utils.read_meta_datasets(
        train_cities=args.train_cities,
        test_city=args.data,
        path=args.datapath,
        tod_embedding_dim=args.tod_embedding_dim,
        dow_embedding_dim=args.dow_embedding_dim,
        steps_per_day=args.steps_per_day
    )
    
    # 生成训练、验证和测试集
    splits_dict = {}

    for c in args.train_cities + [args.data]:
        if c == args.data:  # 如果是目标城市
            logger.info(f"Using limited data for target city {c}: 3 days for adaptation")
            adaptation_days = 3
            adaptation_dataset = data_neural[c]['dataset'][:adaptation_days * args.steps_per_day]
            remaining_dataset = data_neural[c]['dataset'][adaptation_days * args.steps_per_day:]
            
            # 适应性训练集划分（仅训练集）
            adaptation_splits = utils.generate_data(adaptation_dataset, args.seq_len, args.horizons, split_ratios=(1.0, 0.0, 0.0))
            splits_dict[c] = adaptation_splits
            
            # 评估集划分（训练:验证:测试 = 7:1:2）
            eval_splits = utils.generate_data(remaining_dataset, args.seq_len, args.horizons, split_ratios=(0.7, 0.1, 0.2))
            splits_dict[f"{c}_eval"] = eval_splits
        else:
            dataset = data_neural[c]['dataset']
            splits = utils.generate_data(dataset, args.seq_len, args.horizons, split_ratios=(0.7, 0.1, 0.2))
            splits_dict[c] = splits

    # 初始化数据加载器
    train_loaders = {}
    val_loaders = {}
    test_loaders = {}

    for c in args.train_cities:
        train_loaders[c] = utils.get_dataloader(splits_dict[c], args.batch_size, split='train', shuffle=True, horizons=args.horizons)
        val_loaders[c] = utils.get_dataloader(splits_dict[c], args.batch_size, split='val', shuffle=False, horizons=args.horizons)
        test_loaders[c] = utils.get_dataloader(splits_dict[c], args.batch_size, split='test', shuffle=False, horizons=args.horizons)

    # 针对目标城市的适应性训练和评估
    target_city = args.data
    train_loaders[target_city] = utils.get_dataloader(splits_dict[target_city], args.batch_size, split='train', shuffle=True, horizons=args.horizons)
    val_loaders[f"{target_city}_eval"] = utils.get_dataloader(splits_dict[f"{target_city}_eval"], args.batch_size, split='val', shuffle=False, horizons=args.horizons)
    test_loaders[f"{target_city}_eval"] = utils.get_dataloader(splits_dict[f"{target_city}_eval"], args.batch_size, split='test', shuffle=False, horizons=args.horizons)

    # 检查目标城市的评估集是否为空
    if len(splits_dict[f"{target_city}_eval"]['test']) == 0:
        logger.error(f"Target city {target_city} evaluation test set is empty.")
        sys.exit(1)

    all_parts_idx_info = {}
    blocknum_per_city = {}
    for c in args.train_cities + [args.data]:
        coords_path = os.path.join(args.datapath, c, 'coords.npy')
        if not os.path.exists(coords_path):
            logger.warning(f"Coords file not found for city {c}, skip KDTree for it.")
            all_parts_idx_info[c] = (None, None, None)
            continue
        coords = np.load(coords_path) # shape (N,2), coords[:,0]:lng, coords[:,1]:lat
        locations = coords.T # (2,N)
        # 假设recur_times=2，表示递归深度，可根据需求调整
        recur_times = 2
        parts_idx, mxlen = utils.kdTree(locations, recur_times, 0)

        # 使用训练数据构建全局相似度矩阵
        dataset = data_neural[c]['dataset']
        adj = utils.construct_adj(dataset, dataset.shape[1])

        # 计算 blocknum
        num_nodes = dataset.shape[1]
        blocksize = args.blocksize
        factors = args.factors
        blocknum = math.ceil(num_nodes / blocksize)  # 动态设置
        blocknum_per_city[c] = blocknum

        # 使用 reorderData
        sps = blocksize * factors
        ori_parts_idx, reo_parts_idx, reo_all_idx = utils.reorderData(parts_idx, mxlen, adj, sps)
        all_parts_idx_info[c] = (ori_parts_idx, reo_parts_idx, reo_all_idx)

    # 确定 meta_model 的 blocknum 为所有城市中最大的 blocknum
    blocknum_max = max(blocknum_per_city.values())
    logger.info(f"Maximum blocknum across cities: {blocknum_max}")
    
    # 配置元模型
    meta_model_args = dict(
        seed=args.seed,
        data="",
        datapath="",
        domain_specific_params=args.domain_specific_params,
        n_linear=args.n_linear,
        seq_len=args.seq_len,
        horizons=args.horizons,
        num_nodes=meta_num_nodes,
        node_dim=2,
        device=device,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
        meta_lr=args.meta_lr,
        update_lr=args.update_lr,
        meta_epochs=args.meta_epochs,
        city_epochs=args.city_epochs,
        test_epochs=args.test_epochs,
        input_dim=args.input_dim,
        tod_embedding_dim=args.tod_embedding_dim,
        dow_embedding_dim=args.dow_embedding_dim,
        spatial_embedding_dim=args.spatial_embedding_dim,
        adaptive_embedding_dim=args.adaptive_embedding_dim,
        steps_per_day=args.steps_per_day,
        output_dim=args.output_dim,
        temporal_layers=args.temporal_layers,
        spatial_layers=args.spatial_layers,
        blocksize=args.blocksize,
        blocknum=blocknum_max,  # 设置为最大 blocknum
        factors=args.factors
    )
    meta_model = DASA(DASAConfig(**meta_model_args)).to(device)

    # 初始化城市模型和优化器
    model_dict = {}
    optim_dict = {}
    for c in args.train_cities + [args.data]:
        # 动态计算 blocknum
        blocknum = blocknum_per_city[c]
        
        city_args = dict(
            seed=args.seed,
            data=c,
            datapath=args.datapath,
            domain_specific_params=args.domain_specific_params,
            n_linear=args.n_linear,
            seq_len=args.seq_len,
            horizons=args.horizons,
            num_nodes=data_neural[c]['dataset'].shape[1],
            node_dim=2,
            device=device,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
            bias=args.bias,
            meta_lr=args.meta_lr,
            update_lr=args.update_lr,
            meta_epochs=args.meta_epochs,
            city_epochs=args.city_epochs,
            test_epochs=args.test_epochs,
            input_dim=args.input_dim,
            tod_embedding_dim=args.tod_embedding_dim,
            dow_embedding_dim=args.dow_embedding_dim,
            spatial_embedding_dim=args.spatial_embedding_dim,
            adaptive_embedding_dim=args.adaptive_embedding_dim,
            steps_per_day=args.steps_per_day,
            output_dim=args.output_dim,
            temporal_layers=args.temporal_layers,
            spatial_layers=args.spatial_layers,
            blocksize=args.blocksize,
            blocknum=blocknum,  # 动态设置
            factors=args.factors
        )
        model = DASA(DASAConfig(**city_args)).to(device)
        model_dict[c] = model
        optim_dict[c] = model.configure_optimizers(
            weight_decay=0,
            learning_rate=args.update_lr,
            betas=(0.9, 0.999),
            device_type='cuda'
        )
    # scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype != 'float32'))
    scaler = torch.amp.GradScaler(enabled=(args.dtype != 'float32'))

    best_val_res = {h: {'MAE': float('inf'), 'MSE': float('inf'), 'RMSE': float('inf'), 'MAPE': float('inf')} for h in args.horizons}
    
    target_city = args.data

    # 开始元训练循环
    for m_epoch in range(args.meta_epochs):
        logger.info(f"========== Meta Epoch {m_epoch+1}/{args.meta_epochs} ==========")
        for c in args.train_cities:
            # 对源城市进行元训练
            meta_model.copy_invariant_params(model_dict[c])
            model_dict[c].train()

            tra_loader = train_loaders[c]
            ori_parts_idx, reo_parts_idx, reo_all_idx = all_parts_idx_info[c]

            for c_epoch in range(args.city_epochs):
                logger.info(f"MetaEpoch: {m_epoch+1}, City: {c}, CityEpoch: {c_epoch+1}/{args.city_epochs}")
                for batch_idx, (x_seq, y_seq) in enumerate(tra_loader):
                    x_seq, y_seq = x_seq.to(device), {h: y_seq[h].to(device) for h in args.horizons}
                    with args.ctx if args.ctx else torch.enable_grad():
                        optim_dict[c].zero_grad(set_to_none=True)
                        preds, loss = model_dict[c](x_seq, y_seq, train=True,
                                                    ori_parts_idx=ori_parts_idx, 
                                                    reo_parts_idx=reo_parts_idx, 
                                                    reo_all_idx=reo_all_idx)
                        if torch.isnan(loss):
                            logger.error("Loss is NaN. Stopping training.")
                            sys.exit(1)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optim_dict[c])
                        if torch.isnan(loss):
                            logger.error("Gradients are NaN. Stopping training.")
                            sys.exit(1)
                        nn.utils.clip_grad_norm_(model_dict[c].parameters(), args.grad_clip)
                        scaler.step(optim_dict[c])
                        scaler.update()
                        del preds, loss
                        torch.cuda.empty_cache()

            # 验证源城市
            val_loader = val_loaders[c]
            model_dict[c].eval()
            with torch.no_grad():
                v_preds = model_dict[c].evaluate(val_loader)
                for h in args.horizons:
                    mae = v_preds[h]['MAE']
                    mape = v_preds[h]['MAPE']
                    mse = v_preds[h]['MSE']
                    rmse = v_preds[h]['RMSE']
                    logger.info(f"MetaEpoch: {m_epoch+1}, Name: {c}, Horizon: {h}, Val MAE: {mae:.5f}, MAPE: {mape:.2f}%, MSE: {mse:.5f}, RMSE: {rmse:.5f}")

            # 更新元模型的非领域特定参数
            meta_model.eval()
            for name, param in model_dict[c].named_parameters():
                if any(sub_str in name for sub_str in meta_model.config.domain_specific_params):
                    continue
                if param.grad is not None:
                    # 简化的元更新，减去 meta_lr * grad
                    param.data -= args.meta_lr * param.grad

        # 将元模型参数拷贝至目标城市模型
        meta_model.copy_invariant_params(model_dict[target_city])

        # 目标城市的适应性训练
        model_dict[target_city].train()
        tar_loader = train_loaders[target_city]
        ori_parts_idx, reo_parts_idx, reo_all_idx = all_parts_idx_info[target_city]

        for t_epoch in range(args.test_epochs):
            logger.info(f"MetaEpoch: {m_epoch+1}, Target City: {target_city}, TestEpoch: {t_epoch+1}/{args.test_epochs}")
            current_epoch = m_epoch * args.test_epochs + t_epoch
            total_epochs = args.meta_epochs * args.test_epochs
            target_weight = current_epoch / total_epochs
            for batch_idx, (x_seq, y_seq) in enumerate(tar_loader):
                x_seq, y_seq = x_seq.to(device), {h: y_seq[h].to(device) for h in args.horizons}

                with args.ctx if args.ctx else torch.enable_grad():
                    optim_dict[target_city].zero_grad(set_to_none=True)
                    preds, loss = model_dict[target_city](x_seq, y_seq, train=True,
                                                          ori_parts_idx=ori_parts_idx, 
                                                          reo_parts_idx=reo_parts_idx, 
                                                          reo_all_idx=reo_all_idx)
                    weighted_loss = target_weight * loss
                scaler.scale(weighted_loss).backward()
                scaler.unscale_(optim_dict[target_city])
                nn.utils.clip_grad_norm_(model_dict[target_city].parameters(), args.grad_clip)
                scaler.step(optim_dict[target_city])
                scaler.update()
                del preds, loss, weighted_loss
                torch.cuda.empty_cache()

            # 验证目标城市（使用评估集）
            val_loader = val_loaders[f"{target_city}_eval"]
            test_loader = test_loaders[f"{target_city}_eval"]
            model_dict[target_city].eval()
            with torch.no_grad():
                val_results = model_dict[target_city].evaluate(val_loader)

                for h in args.horizons:
                    logger.info(
                        f"Test_epoch {t_epoch+1}, Horizon {h}: "
                        f"Val MAE {val_results[h]['MAE']:.5f}, Val MAPE {val_results[h]['MAPE']:.2f}%, "
                        f"Val MSE {val_results[h]['MSE']:.5f}, Val RMSE {val_results[h]['RMSE']:.5f}"
                    )

                    # 更新best_val_res
                    if (val_results[h]['MAE'] < best_val_res[h]['MAE'] and
                        val_results[h]['RMSE'] < best_val_res[h]['RMSE']):
                        best_val_res[h]['MAE'] = val_results[h]['MAE']
                        best_val_res[h]['MAPE'] = val_results[h]['MAPE']
                        best_val_res[h]['MSE'] = val_results[h]['MSE']
                        best_val_res[h]['RMSE'] = val_results[h]['RMSE']
                        if m_epoch > 0 or t_epoch > 0:  # 保存至少在第一个epoch后
                            ckpt = {
                                'model': model_dict[target_city].state_dict(),
                                'optimizer': optim_dict[target_city].state_dict(),
                                'model_args': city_args,
                                'm_epoch': m_epoch,
                                't_epoch': t_epoch,
                                'best_val_res': best_val_res
                            }
                            logger.info(f"Saving checkpoint to {args.out_dir}")
                            torch.save(ckpt, path_test_model)

    # 保存元模型和最后一个模型检查点
    torch.save({'state_dict': meta_model.state_dict()}, path_meta_model)
    ckpt_last = {
        'model': model_dict[args.data].state_dict(),
        'optimizer': optim_dict[args.data].state_dict(),
        'model_args': city_args,
        'm_epoch': m_epoch,
        't_epoch': t_epoch,
        'best_val_res': best_val_res
    }
    torch.save(ckpt_last, path_test_model_last)

    logger.info("========== Evaluation on Adaptation Validation Set ==========")
    final_val_loader = val_loaders[f"{target_city}_eval"]
    model_dict[target_city].eval()
    with torch.no_grad():
        final_val_results = model_dict[target_city].evaluate(final_val_loader)
        logger.info("========== Adaptation Validation Evaluate results ==========")
        for h in args.horizons:
            mae = final_val_results[h]['MAE']
            mape = final_val_results[h]['MAPE']
            mse = final_val_results[h]['MSE']
            rmse = final_val_results[h]['RMSE']
            logger.info(f"Horizon {h}: Adapt Val MAE={mae:.3f}, Adapt Val MAPE={mape:.3f}%, Adapt Val MSE={mse:.3f}, Adapt Val RMSE={rmse:.3f}")

        final_val_results_str_keys = {str(k): v for k, v in final_val_results.items()}
        np.savez(os.path.join(args.out_dir, f'adapt_val_results_{args.data}.npz'), **final_val_results_str_keys)

    # 仅保留对测试集的最终评估
    logger.info("========== Final Evaluation on Test Set ==========")
    test_loader = test_loaders[f"{target_city}_eval"]
    model_dict[target_city].eval()
    with torch.no_grad():
        test_results = model_dict[target_city].evaluate(test_loader)
        logger.info("========== Test Results ==========")
        for h in args.horizons:
            mae = test_results[h]['MAE']
            mape = test_results[h]['MAPE']
            mse = test_results[h]['MSE']
            rmse = test_results[h]['RMSE']
            logger.info(f"Horizon {h}: MAE={mae:.3f}, MAPE={mape:.3f}%, MSE={mse:.3f}, RMSE={rmse:.3f}")

    test_results_str_keys = {str(k): v for k, v in test_results.items()}
    np.savez(os.path.join(args.out_dir, f'test_results_{args.data}.npz'), **test_results_str_keys)
