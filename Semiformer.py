import math
import inspect
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
import sys
import torch.nn.init as init

# 获取当前模块的logger
logger = logging.getLogger(__name__)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init.ones_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class MLP(nn.Module):
    def __init__(self, config, final_dim):
        super().__init__()
        # 输入维final_dim，输出4*final_dim
        self.c_fc = nn.Linear(final_dim, 4 * final_dim, bias=config.bias)
        self.c_proj = nn.Linear(4 * final_dim, final_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class SemiSelfAttention(nn.Module):
    """
    原始的时间或空间维度的因果自注意力机制（SemiSelfAttention）。
    Q,K 来自h_shared，V来自h_private。
    """
    def __init__(self, config, final_dim):
        super().__init__()
        self.key_mlp = nn.Linear(final_dim, final_dim, bias=True)
        self.query_mlp = nn.Linear(final_dim, final_dim, bias=True)
        self.value_mlp = nn.Linear(final_dim, final_dim, bias=True)
        self.value_transform = nn.Linear(final_dim, final_dim, bias=config.bias)

        self.c_proj = nn.Linear(final_dim, final_dim, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, h_private, h_shared, attn_dim=-2):
        def move_attn_dim(x):
            dims = list(range(x.dim()))
            if attn_dim != -2:
                dims.remove(attn_dim)
                dims.insert(-2, attn_dim)
                x = x.permute(*dims)
            return x, dims

        h_private_moved, dims_p = move_attn_dim(h_private)
        h_shared_moved, dims_s = move_attn_dim(h_shared)

        B = h_shared_moved.shape[0]
        L = h_shared_moved.shape[-2]
        C = h_shared_moved.shape[-1]

        v = self.value_mlp(h_private_moved)
        v = self.value_transform(v)
        q = self.query_mlp(h_shared_moved)
        k = self.key_mlp(h_shared_moved)

        def split_heads(t):
            return t.view(*t.shape[:-1], self.n_head, C // self.n_head).transpose(-3, -2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(-3, -2).contiguous().view(*y.shape[:-3], L, C)
        y = self.resid_dropout(self.c_proj(y))

        inv_dims = [0]*len(dims_p)
        for i, d in enumerate(dims_p):
            inv_dims[d] = i
        y = y.permute(*inv_dims)

        return y

class Block(nn.Module):
    """
    Block接受(h_private, h_shared)输入，并进行attention和mlp操作
    """
    def __init__(self, config, final_dim):
        super().__init__()
        self.ln_1_priv = LayerNorm(final_dim, bias=config.bias)
        self.ln_1_shared = LayerNorm(final_dim, bias=config.bias)
        self.attn = SemiSelfAttention(config, final_dim=final_dim)
        self.ln_2_priv = LayerNorm(final_dim, bias=config.bias)
        self.mlp_priv = MLP(config, final_dim=final_dim)
        # 若需要对shared也处理，可新增mlp_shared层

    def forward(self, h_private, h_shared, attn_dim):
        # LayerNorm
        h_private_ln = self.ln_1_priv(h_private)
        h_shared_ln = self.ln_1_shared(h_shared)

        # 注意力计算：q,k来自shared_ln, v来自private_ln
        attn_output = self.attn(h_private_ln, h_shared_ln, attn_dim=attn_dim)

        # 将注意力输出加回private分支
        h_private = h_private + attn_output
        h_private = h_private + self.mlp_priv(self.ln_2_priv(h_private))

        return h_private, h_shared


class DualAttBlock(nn.Module):
    
    def __init__(self, hidden_size, num_heads, P, N, mlp_ratio=4.0, dropout=0.1, bias=True):
        super().__init__()
        self.num = P
        self.size = N

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.dropout = dropout

        # Intra-Block Attention
        self.snorm1 = LayerNorm(hidden_size, bias=bias)
        self.sattn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, bias=True)
        self.snorm2 = LayerNorm(hidden_size, bias=bias)
        self.smlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout)
        )

        # Inter-Block Attention
        self.nnorm1 = LayerNorm(hidden_size, bias=bias)
        self.nattn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, bias=True)
        self.nnorm2 = LayerNorm(hidden_size, bias=bias)
        self.nmlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x_private, x_shared):
        # x_private, x_shared: [B,T,N,D]
        B, T, N, D = x_private.shape
        P, S = self.num, self.size
        assert P * S >= N, "P*S must be >= N"

        if P * S > N:
            pad_size = P * S - N
            pad_tensor = torch.zeros(B, T, pad_size, D).to(x_private.device)
            x_private = torch.cat([x_private, pad_tensor], dim=2)  # (B, T, P*S, D)
            x_shared = torch.cat([x_shared, pad_tensor], dim=2)    # (B, T, P*S, D)
            N_padded = P * S  # 更新N

        else:
            N_padded = N

        # Intra-Attention
        # reshape为(B*T*P, S, D)
        x_private_reshaped = x_private.reshape(B * T * P, S, D)
        x_shared_reshaped = x_shared.reshape(B * T * P, S, D)

        # Q,K来自shared，V来自private
        # 先LayerNorm
        qs = self.snorm1(x_shared_reshaped)  # Q,K
        vs = self.snorm1(x_private_reshaped)  # V 使用与Q,K相同的归一化处理，但实际V来自private表示
        qs = qs.transpose(0, 1)  # (S, B*T*P, D)
        ks = qs
        vs = vs.transpose(0, 1)  # (S, B*T*P, D)

        attn_out, _ = self.sattn(qs, ks, vs)
        attn_out = attn_out.transpose(0, 1)  # (B*T*P, S, D)
        x_private_reshaped = x_private_reshaped + attn_out
        x_private_reshaped = x_private_reshaped + self.smlp(self.snorm2(x_private_reshaped))

        # Inter-Block Attention
        # 将 (B, T, P, S, D) 转换为 (B*T*S, P, D)
        x_private_local = x_private_reshaped.reshape(B, T, P, S, D).transpose(2, 3).reshape(B * T * S, P, D)
        x_shared_local = x_shared_reshaped.reshape(B, T, P, S, D).transpose(2, 3).reshape(B * T * S, P, D)

        qb = self.nnorm1(x_shared_local)  # Q,K from shared
        vb = self.nnorm1(x_private_local)  # V from private
        qb = qb.transpose(0, 1)  # (P, B*T*S, D)
        kb = qb
        vb = vb.transpose(0, 1)  # (P, B*T*S, D)
        attn_out_breadth, _ = self.nattn(qb, kb, vb)
        attn_out_breadth = attn_out_breadth.transpose(0, 1)  # (B*T*S, P, D)

        x_private_local = x_private_local + attn_out_breadth
        x_private_local = x_private_local + self.nmlp(self.nnorm2(x_private_local))

        # 重塑回原形状 (B*T*S, P, D) -> (B, T, S, P, D) -> (B, T, P * S, D) -> (B, T, N, D)
        x_private_final = x_private_local.reshape(B, T, S, P, D).transpose(2, 3).reshape(B, T, P * S, D)

        if P * S > N:
            # 去掉填充部分
            x_private_final = x_private_final[:, :, :N, :]
            x_shared_final = x_shared[:, :, :N, :]  # 同样去掉填充
        else:
            x_shared_final = x_shared

        return x_private_final, x_shared_final


@dataclass
class DASAConfig:
    seed: int = 0
    data: str = ''
    datapath: str = ''
    seq_len: int = 12
    horizons: list = field(default_factory=lambda: [12])
    num_nodes: int = 325
    node_dim: int = 2
    device: torch.device = torch.device('cuda:0')
    n_linear: int = 1
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.1
    bias: bool = True
    meta_lr: float = 1e-3
    update_lr: float = 0.01
    meta_epochs: int = 5
    city_epochs: int = 2
    test_epochs: int = 50
    domain_specific_params: list = field(default_factory=lambda: ['value_mlp', 'value_transform'])

    input_dim: int = 2        # 原始输入特征维度（例如速度+其他1维特征）
    tod_embedding_dim: int = 12
    dow_embedding_dim: int = 6
    spatial_embedding_dim: int =6 
    adaptive_embedding_dim: int = 36
    steps_per_day: int = 288  # 假设的每日步数，可根据数据修改
    output_dim: int = 1       # 输出预测特征维度，一般为1（预测速度）
    temporal_layers: int = 1
    spatial_layers: int = 1
    
    blocksize: int = 8
    blocknum: int = 4
    factors: int = 1
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class DASAEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.n_embd)

        if config.tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(config.steps_per_day, config.tod_embedding_dim)
        else:
            self.tod_embedding = None

        if config.dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, config.dow_embedding_dim)
        else:
            self.dow_embedding = None

        if config.spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(torch.empty(config.num_nodes, config.spatial_embedding_dim))
            nn.init.xavier_uniform_(self.node_emb)
        else:
            self.node_emb = None

        if config.adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(torch.empty(config.seq_len, config.num_nodes, config.adaptive_embedding_dim))
            nn.init.xavier_uniform_(self.adaptive_embedding)
        else:
            self.adaptive_embedding = None

        self.model_dim = (config.n_embd
                          + config.tod_embedding_dim
                          + config.dow_embedding_dim
                          + config.spatial_embedding_dim
                          + config.adaptive_embedding_dim)

    def forward(self, x):
        B,T,N,C = x.size()

        features = []

        # 基础input投射
        main_input = x[..., :self.config.input_dim] # (B,T,N,input_dim)
        main_input = self.input_proj(main_input)    # (B,T,N,n_embd)
        features.append(main_input)

        # tod embedding
        if self.tod_embedding is not None and C > self.config.input_dim:
            # 假设 x[...,self.config.input_dim] 为tod特征(0~1映射到steps_per_day)
            tod_idx = (x[...,self.config.input_dim]*self.config.steps_per_day).long().clamp(0, self.config.steps_per_day-1)
            tod_emb = self.tod_embedding(tod_idx)   # (B,T,N,tod_embedding_dim)
            features.append(tod_emb)

        # dow embedding
        # 若有dow特征则应位于 x[...,self.config.input_dim+1]
        if self.dow_embedding is not None and C > self.config.input_dim+1:
            dow_idx = x[...,self.config.input_dim+1].long().clamp(0,6)
            dow_emb = self.dow_embedding(dow_idx)  # (B,T,N,dow_embedding_dim)
            features.append(dow_emb)

        # spatial embedding
        if self.node_emb is not None:
            # node_emb: (N, spatial_embedding_dim)
            # 扩展到 (B,T,N,spatial_embedding_dim)
            spatial_emb = self.node_emb.unsqueeze(0).unsqueeze(0).expand(B,T,N,self.config.spatial_embedding_dim)
            features.append(spatial_emb)

        # adaptive embedding
        if self.adaptive_embedding is not None:
            # adaptive_embedding: (T,N,adp_dim)
            # 扩展到(B,T,N,adp_dim)
            adp_emb = self.adaptive_embedding.unsqueeze(0).expand(B,T,N,self.config.adaptive_embedding_dim)
            features.append(adp_emb)

        x = torch.cat(features, dim=-1)  # (B,T,N,model_dim)

        return x


class DASA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 引入新Embedding层
        self.embedding = DASAEmbedding(config)

        # temporal_layers和spatial_layers
        # 使用Block作为时间Transformer单元
        final_dim = (config.n_embd 
             + config.tod_embedding_dim 
             + config.dow_embedding_dim 
             + config.spatial_embedding_dim 
             + config.adaptive_embedding_dim)

         # private and shared projection
        self.proj_private = nn.Linear(final_dim, final_dim, bias=True)
        self.proj_shared = nn.Linear(final_dim, final_dim, bias=True)
        
        # 时间Transformer block列表
        self.temporal_blocks = nn.ModuleList([Block(config, final_dim=final_dim) for _ in range(config.temporal_layers)])
        
        # 空间Transformer block（使用DualAttBlock）
        self.spa_num = config.blocknum
        self.spa_size = config.blocksize
        self.factors = config.factors
        self.spatial_att_block = DualAttBlock(
            hidden_size=final_dim,
            num_heads=config.n_head,
            P=self.spa_num//self.factors,
            N=self.spa_size*self.factors,
            mlp_ratio=1.0,  #可根据需求调节
            dropout=config.dropout,
            bias=config.bias
        )
        

        self.ln_f = LayerNorm(final_dim, bias=config.bias)
        self.outs = nn.ModuleDict({
            str(h): nn.Linear(final_dim, h * self.config.output_dim) for h in config.horizons
        })

        logger.info("number of parameters: %.2fM", self.get_num_params() /1e6)
        initialize_weights(self)
        
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def forward(self, x, y=None, train=True, ori_parts_idx=None, reo_parts_idx=None, reo_all_idx=None):
        # x: (B,T,N,C)
        # 在进入并行路径前进行Embedding
        x_embed = self.embedding(x) # (B,T,N,D)
        
        # 分为private与shared
        h_private = self.proj_private(x_embed) # (B,T,N,D)
        h_shared = self.proj_shared(x_embed)   # (B,T,N,D)
        
        # 时间路径
        for block in self.temporal_blocks:
            h_private, h_shared = block(h_private, h_shared, attn_dim=1)

        # 空间路径
        h_private, h_shared = self.spatial_att_block(h_private, h_shared)

        # 融合时间路径和空间路径的输出
        h_final = h_private + h_shared

        h_final = self.ln_f(h_final)
        # 取最后时间步 (B,N,D)
        h_final = h_final[:, -1, :, :]

        preds = {}
        for h in self.config.horizons:
            ph = self.outs[str(h)](h_final)  # (B,N,h*output_dim)
            ph = ph.view(ph.size(0), ph.size(1), h*self.config.output_dim)  # (B,N,h)
            ph = ph.permute(0,2,1).contiguous() # (B,h,N)
            preds[h] = ph

        if train and y is not None:
            loss = 0
            for h in self.config.horizons:
                loss += self.loss(preds[h], y[h])
            if torch.isnan(loss):
                logger.error("Loss is NaN. Stopping training.")
                sys.exit(1)
            return preds, loss
        else:
            return preds

    def loss(self, pred, target):
        return nn.L1Loss()(pred, target) + nn.MSELoss()(pred, target)

    def copy_invariant_params(self, city_model):
        for m_name, m_param in self.named_parameters():
            # 如果参数在domain_specific_params中，则跳过
            if any(sub_str in m_name for sub_str in self.config.domain_specific_params):
                logger.debug("Skipping domain-specific parameter: %s", m_name)
                continue
            if m_name in city_model.state_dict():
                c_param = city_model.state_dict()[m_name]
                if m_param.shape == c_param.shape:
                    city_model.state_dict()[m_name].copy_(m_param)
                    logger.debug("Copied parameter: %s", m_name)
                else:
                    logger.debug("Skipping copying parameter %s due to shape mismatch: %s vs %s", m_name, m_param.shape, c_param.shape)
            else:
                logger.debug("Parameter %s not found in city model.", m_name)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [{"params": list(param_dict.values()), "weight_decay": weight_decay}]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer
    
    @torch.no_grad()
    def evaluate(self, data):
        self.eval()
        preds = {h: [] for h in self.config.horizons}
        targets = {h: [] for h in self.config.horizons}
        for x, y in data:
            x = x.to(self.config.device)
            y = {h: y[h].to(self.config.device) for h in self.config.horizons}
            pred = self(x, train=False)
            for h in self.config.horizons:
                preds[h].append(pred[h].cpu().detach().numpy())
                targets[h].append(y[h].cpu().detach().numpy())
        for h in self.config.horizons:
            preds[h] = np.concatenate(preds[h], axis=0)
            targets[h] = np.concatenate(targets[h], axis=0)
        results = {}
        for h in self.config.horizons:
            mae = np.mean(np.abs(preds[h] - targets[h]))
            mse = np.mean((preds[h] - targets[h]) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((targets[h] - preds[h]) / (targets[h] + 1e-8))) * 100
            results[h] = {'MAE': mae, 'MAPE': mape, 'MSE': mse, 'RMSE': rmse}
        return results
