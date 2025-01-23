
import torch
import numpy as np
import random
import logging
from datetime import datetime
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from sklearn.metrics.pairwise import cosine_similarity

CITY_DICT = {
    'chengdu': 'Chengdu',
    'shenzhen': 'Shenzhen',
    'pems-bay': 'PemsBay',
    'metr-la': 'MetrLA'
}

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_logger(log_dir='./logs/', log_prefix=''):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)-8s %(message)s',
            "%Y-%m-%d %H:%M:%S")
        
        # 控制台日志
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        
        # 文件日志
        ts = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        fh = RotatingFileHandler(f'{log_dir}/{log_prefix}-{ts}.log', maxBytes=10*1024*1024, backupCount=5)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

def read_meta_datasets(train_cities=[], test_city='', path='',
                       tod_embedding_dim=24, dow_embedding_dim=8, steps_per_day=288):
    """
    读取元数据集，并根据需求添加tod/dow特征。
    
    参数:
    - train_cities: list，训练城市列表
    - test_city: str，目标城市名称
    - path: str，数据文件的路径
    - tod_embedding_dim: int, 若>0则添加tod特征 (0~1)
    - dow_embedding_dim: int, 若>0则添加dow特征 (0~6)
    - steps_per_day: int, 每天的时间步数，用于计算tod/dow

    返回:
    - data_neural: dict，每个城市的数据 {city: {'dataset': np.array, 'adj_mat': np.array}}
    - num_nodes: int，节点总数
    """
    data_neural = {c: {} for c in train_cities + [test_city]}
    num_nodes = 0
    for c in train_cities + [test_city]:
        dataset = np.load(f'{path}/{c}/dataset.npy') # (T, N, C)
        adj_mat = np.load(f'{path}/{c}/matrix.npy')

        T, N, C = dataset.shape
        time_idx = np.arange(T)

        new_features = []
        # TOD特征: (time_idx % steps_per_day) / steps_per_day, 范围0~1
        if tod_embedding_dim > 0:
            tod_feature = ((time_idx % steps_per_day) / steps_per_day).reshape(T,1) # (T,1)
            tod_feature = np.tile(tod_feature, (1, N))[:,:,None] # (T,N,1)
            new_features.append(tod_feature)

        # DOW特征: (time_idx // steps_per_day) % 7
        if dow_embedding_dim > 0:
            dow_feature = ((time_idx // steps_per_day) % 7).reshape(T,1) # (T,1)
            dow_feature = np.tile(dow_feature, (1, N))[:,:,None] # (T,N,1)
            new_features.append(dow_feature)

        if len(new_features) > 0:
            # 将新特征拼接到dataset最后一维
            new_features_all = np.concatenate(new_features, axis=-1) 
            dataset = np.concatenate([dataset, new_features_all], axis=-1)

        data_neural[c]['dataset'] = dataset
        data_neural[c]['adj_mat'] = adj_mat
        num_nodes = max(num_nodes, dataset.shape[1])
    return data_neural, num_nodes



def generate_data(dataset, seq_len, horizons, split_ratios=(0.7, 0.1, 0.2)):
    """
    生成多时间步的预测数据，并划分为训练集、验证集和测试集。

    参数:
    - dataset: numpy 数组，形状为 [时间步, 节点数, 特征数]
    - seq_len: 输入序列长度
    - horizons: list，包含多个预测时间步
    - split_ratios: tuple，训练集、验证集、测试集的比例

    返回:
    - splits: dict，包含 'train', 'val', 'test' 三个键，每个键对应输入数据和预测数据
    """
    x, ys = [], {h: [] for h in horizons}
    max_horizon = max(horizons)
    T = len(dataset)
    for i in range(T - seq_len - max_horizon):
        a = dataset[i: i + seq_len, :, :]  # (seq_len, N, C)
        skip = False
        for h in horizons:
            b = dataset[i + seq_len: i + seq_len + h, :, 0]  # 预测目标为第0维特征
            if np.isnan(a).any() or np.isnan(b).any() or np.isinf(a).any() or np.isinf(b).any():
                skip = True
                break
            ys[h].append(b)
        if not skip:
            x.append(a)

    if not x:
        splits = {
            'train': {'x': np.array([]), 'y': {h: np.array([]) for h in horizons}},
            'val': {'x': np.array([]), 'y': {h: np.array([]) for h in horizons}},
            'test': {'x': np.array([]), 'y': {h: np.array([]) for h in horizons}}
        }
        return splits

    x = np.stack(x, axis=0)  # (num_samples, seq_len, N, C)
    for h in horizons:
        ys[h] = np.stack(ys[h], axis=0)  # (num_samples, h, N)

    total_samples = x.shape[0]

    if len(split_ratios) == 2:
        split_ratios = (split_ratios[0], split_ratios[1], 0.0)
    elif len(split_ratios) == 1:
        split_ratios = (split_ratios[0], 0.0, 0.0)
    elif len(split_ratios) > 3:
        raise ValueError("split_ratios 的长度不能超过3。")

    train_ratio, val_ratio, test_ratio = split_ratios
    assert train_ratio + val_ratio + test_ratio <= 1.0, "split_ratios 的总和不能超过1.0"

    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    splits = {}
    if train_ratio > 0:
        splits['train'] = {
            'x': x[:train_end],
            'y': {h: ys[h][:train_end] for h in horizons}
        }
    else:
        splits['train'] = {'x': np.array([]), 'y': {h: np.array([]) for h in horizons}}

    if val_ratio > 0:
        splits['val'] = {
            'x': x[train_end:val_end],
            'y': {h: ys[h][train_end:val_end] for h in horizons}
        }
    else:
        splits['val'] = {'x': np.array([]), 'y': {h: np.array([]) for h in horizons}}

    if test_ratio > 0:
        splits['test'] = {
            'x': x[val_end:],
            'y': {h: ys[h][val_end:] for h in horizons}
        }
    else:
        splits['test'] = {'x': np.array([]), 'y': {h: np.array([]) for h in horizons}}

    return splits



def get_dataloader(splits, batch_size, split='train', shuffle=True, drop_last=False, horizons=[12]):
    """
    根据数据集划分生成数据加载器。

    参数:
    - splits: dict，包含 'train', 'val', 'test' 三个子集的数据
    - batch_size: int，批量大小
    - split: str，选择 'train'、'val' 或 'test'
    - shuffle: bool，是否打乱数据
    - drop_last: bool，是否丢弃最后一个不足批量的数据
    - horizons: list，预测步长列表

    返回:
    - data_loader: list，包含批量数据的列表，每个元素为 (batch_x, batch_y)
    """
    split_data = splits[split]
    x, y_tests = split_data['x'], split_data['y']
    data_len = x.shape[0]
    num_batches = data_len // batch_size
    if not drop_last:
        num_batches += int((data_len % batch_size) != 0)
    data_loader = []
    indices = list(range(data_len))
    if shuffle and split == 'train':
        np.random.shuffle(indices)
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, data_len)
        batch_index = indices[start_index:end_index]
        batch_x = torch.from_numpy(x[batch_index]).float()
        batch_y = {h: torch.from_numpy(y_tests[h][batch_index]).float() for h in horizons}
        data_loader.append((batch_x, batch_y))
    return data_loader


def compute_grad_norm(model):
    """
    计算模型的梯度范数，用于梯度裁剪。
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)

def compute_mape(pred, target):
    """
    计算 MAPE 指标。
    """
    epsilon = 1e-8
    return torch.mean(torch.abs((target - pred) / (target + epsilon))) * 100


def construct_adj(data, num_node):
    """
    基于余弦相似度构建邻接矩阵

    参数:
    - data: 时间序列数据 (T, N, C)
    - num_node: 节点数
    
    返回:
    - tem_matrix: 相似度邻接矩阵 (N, N)
    """
    steps_per_day = 288
    days_count = data.shape[0] // steps_per_day
    if days_count == 0:
        # 对整段数据求均值 (T, N, C) -> (N, C)
        data_mean = data.mean(axis=0)  # (N, C)
    else:
        # 取前 days_count * steps_per_day 进行均值 (days_count * steps_per_day, N, C) -> (N, C)
        data_subset = data[:days_count * steps_per_day]
        data_mean = data_subset.mean(axis=0)  # (N, C)
    
    # 计算余弦相似度矩阵 (N, C) 和 (N, C) -> (N, N)
    tem_matrix = cosine_similarity(data_mean, data_mean)
    tem_matrix = np.exp((tem_matrix - tem_matrix.mean()) / tem_matrix.std())
    return tem_matrix

def augmentAlign(dist_matrix, auglen):
    """
	根据余弦相似度对分块进行填充，选择相似节点填充分块
    从dist_matrix中选择与已有点最相似的auglen个点进行填充
    dist_matrix: (n, n), n为节点数
    auglen: 填充数量
    """
    sorted_idx = np.argsort(dist_matrix.reshape(-1)*-1)
    sorted_idx = sorted_idx % dist_matrix.shape[-1]
    augidx = []
    for idx in sorted_idx:
        if idx not in augidx:
            augidx.append(idx)
        if len(augidx) == auglen:
            break
    return np.array(augidx, dtype=int)

def reorderData(parts_idx, mxlen, adj, sps):
    """
    对分块后的数据进行重排序和填充，使得每个块大小相同并便于并行计算。
    
    参数:
    - parts_idx: KDTree分块后的节点索引列表的列表，例如[[idx_of_block1],[idx_of_block2],...]
    - mxlen: 分块中最大节点数
    - adj: 节点相似度矩阵 (N,N)
    - sps: 每个分块的最终大小（填充后）
    
    返回:
    - ori_parts_idx: 原始节点索引对应的排序
    - reo_parts_idx: 重排序后的分块内的索引对应关系
    - reo_all_idx: 最终的填充后索引，用于对原数据进行重排
    """
    ori_parts_idx = np.array([], dtype=int)
    reo_parts_idx = np.array([], dtype=int)
    reo_all_idx = np.array([], dtype=int)
    for i, part_idx in enumerate(parts_idx):
        part_dist = adj[part_idx, :].copy()
        part_dist[:, part_idx] = 0
        # 若需要填充
        if sps - part_idx.shape[0] > 0:
            local_part_idx = augmentAlign(part_dist, sps - part_idx.shape[0])
            auged_part_idx = np.concatenate([part_idx, local_part_idx], 0)
        else:
            auged_part_idx = part_idx

        reo_parts_idx = np.concatenate([reo_parts_idx, np.arange(part_idx.shape[0]) + sps*i])
        ori_parts_idx = np.concatenate([ori_parts_idx, part_idx])
        reo_all_idx = np.concatenate([reo_all_idx, auged_part_idx])

    return ori_parts_idx, reo_parts_idx, reo_all_idx

def kdTree(locations, times, axis):
    
    sorted_idx = np.argsort(locations[axis])
    half = locations.shape[1] // 2
    part1, part2 = np.sort(sorted_idx[:half]), np.sort(sorted_idx[half:])
    parts = []
    if times == 1:
        return [part1, part2], max(part1.shape[0], part2.shape[0])
    else:
        left_parts, lmxlen = kdTree(locations[:,part1], times-1, axis^1)
        right_parts, rmxlen = kdTree(locations[:,part2], times-1, axis^1)
        for part in left_parts:
            parts.append(part1[part])
        for part in right_parts:
            parts.append(part2[part])
    return parts, max(lmxlen, rmxlen)




