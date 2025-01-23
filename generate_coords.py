# generate_coords.py

import os
import argparse
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from sklearn.decomposition import PCA
from umap import UMAP
import logging

def set_logger(log_file='generate_coords.log'):
    logger = logging.getLogger('generate_coords')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s', "%Y-%m-%d %H:%M:%S")
    
    # 控制台日志
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # 文件日志
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

def check_and_handle_connectivity(adj_mat):
    G = nx.from_numpy_array(adj_mat)
    if not nx.is_connected(G):
        connected_components = list(nx.connected_components(G))
        print(f"Number of connected components: {len(connected_components)}")
        largest_cc = max(connected_components, key=len)
        G = G.subgraph(largest_cc).copy()
        adj_mat = nx.to_numpy_array(G)
    return adj_mat, G

def generate_coords_for_city(city, datapath, embedding_dim=32, dim=2, method='pca', walk_length=100, num_walks=20):
    logger = logging.getLogger('generate_coords')
    adj_path = os.path.join(datapath, city, 'matrix.npy')
    coords_output_path = os.path.join(datapath, city, 'coords.npy')
    
    if not os.path.exists(adj_path):
        logger.warning(f"Adjacency matrix file not found for city {city} at {adj_path}. Skipping.")
        return
    
    # 加载邻接矩阵并处理连通性
    adj_mat = np.load(adj_path)
    adj_mat, G = check_and_handle_connectivity(adj_mat)
    logger.info(f"Loaded adjacency matrix for city {city}, number of nodes: {G.number_of_nodes()}")
    
    # 生成 Node2Vec 嵌入
    node2vec = Node2Vec(G, dimensions=embedding_dim, walk_length=walk_length, num_walks=num_walks, workers=4, quiet=True)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    # 获取嵌入向量
    embeddings = np.zeros((G.number_of_nodes(), embedding_dim))
    for i, node in enumerate(G.nodes()):
        embeddings[i] = model.wv[str(node)]
    
    logger.info(f"Generated Node2Vec embeddings for city {city}, shape: {embeddings.shape}")
    
    # 使用PCA或UMAP降维
    if method == 'umap':
        umap = UMAP(n_components=dim, n_neighbors=15, min_dist=0.1, metric='euclidean')
        coords = umap.fit_transform(embeddings)
        logger.info(f"UMAP transformed embeddings to {dim} dimensions for city {city}, shape: {coords.shape}")
    elif method == 'pca':
        pca = PCA(n_components=dim)
        coords = pca.fit_transform(embeddings)
        logger.info(f"PCA transformed embeddings to {dim} dimensions for city {city}, shape: {coords.shape}")
    
    # 保存coords.npy
    np.save(coords_output_path, coords)
    logger.info(f"Saved coords.npy for city {city} at {coords_output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate coords.npy using Node2Vec for each city.")
    parser.add_argument('--datapath', type=str, required=True, help="Path to the data directory containing city subdirectories.")
    parser.add_argument('--embedding_dim', type=int, default=32, help="Dimension of Node2Vec embeddings.")
    parser.add_argument('--dim', type=int, default=2, help="Dimension to reduce embeddings to.")
    parser.add_argument('--method', type=str, default='pca', choices=['pca', 'umap'], help="Dimensionality reduction method.")
    parser.add_argument('--walk_length', type=int, default=100, help="Length of random walks.")
    parser.add_argument('--num_walks', type=int, default=20, help="Number of random walks per node.")
    parser.add_argument('--cities', nargs='+', default=None, help="List of cities to process. If not provided, all subdirectories in datapath will be processed.")
    
    args = parser.parse_args()
    logger = set_logger()
    
    datapath = args.datapath
    embedding_dim = args.embedding_dim
    dim = args.dim
    method = args.method
    walk_length = args.walk_length
    num_walks = args.num_walks
    
    if args.cities:
        cities = args.cities
    else:
        # 自动发现所有城市子目录
        cities = [name for name in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, name))]
    
    logger.info(f"Starting coordinate generation for cities: {cities}")
    
    for city in cities:
        generate_coords_for_city(city, datapath, embedding_dim, dim, method, walk_length, num_walks)
    
    logger.info("Coordinate generation completed.")

if __name__ == "__main__":
    main()
