import torch, einops
from torch import nn
from torch.nn import functional as F
from mmcv.cnn import ConvModule
from timm.models.layers import DropPath, trunc_normal_

import math
import numpy as np


class Attention(nn.Module):
    '''
    Multi-Head Linear Attention from PvTv2, if linear=True, it will use linear attention.
    '''
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, query, feat, H, W):
        B, N, dim = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, dim, H, W)
                x_ = self.sr(x_).reshape(B, dim, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, dim // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, dim // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, dim, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, dim, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, dim // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x



class CrossAttention(nn.Module):
    def __init__(self, dim=256, num_heads=8):
        """
        Cross Attention Module
        :param dim: dimension  
        :param num_heads: number of attention heads
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError('dim must be divisible by num_heads, but got {} and {}'.format(dim, num_heads))
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Projection layers
        self.query_proj = nn.Linear(dim, dim)  # Projection for query
        self.key_proj = nn.Linear(dim, dim)    # Projection for keys (and values)
        self.value_proj = nn.Linear(dim, dim)  # Projection for values
        self.output_proj = nn.Linear(dim, dim)  # Output projection

    def forward(self, query, input_flatten, input_padding_mask=None):
        """
        :param query: (N, Length_{query}, dim)
        :param input_flatten: (N, Len_in, dim)
        :param input_padding_mask: (N, Len_in), True for padding elements, False for non-padding elements 
        
        :return: output: (N, Length_{query}, dim)
        """
        N, Lq, _ = query.shape
        _, Lin, _ = input_flatten.shape

        # Projection to get queries, keys, and values
        queries = self.query_proj(query)  # (N, Lq, dim)
        keys = self.key_proj(input_flatten)  # (N, Lin, dim)
        values = self.value_proj(input_flatten)  # (N, Lin, dim)

        # Reshape for multi-head attention (N, Lq, num_heads, head_dim)
        queries = queries.view(N, Lq, self.num_heads, self.head_dim).transpose(1, 2)  # (N, num_heads, Lq, head_dim)
        keys = keys.view(N, Lin, self.num_heads, self.head_dim).transpose(1, 2)  # (N, num_heads, Lin, head_dim)
        values = values.view(N, Lin, self.num_heads, self.head_dim).transpose(1, 2)  # (N, num_heads, Lin, head_dim)

        # Calculate attention scores (scaled dot-product attention)
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))  # (N, num_heads, Lq, Lin)
        attn_scores = attn_scores / (self.head_dim ** 0.5)  # Scale by sqrt(head_dim)

        if input_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(input_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)  # (N, num_heads, Lq, Lin)
        attn_output = torch.matmul(attn_weights, values)  # (N, num_heads, Lq, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(N, Lq, self.dim)  # (N, Lq, dim)
        output = self.output_proj(attn_output)  # (N, Lq, dim)

        return output


from einops import rearrange

class VectorizedSparseAttention(nn.Module):
    def __init__(self, dim=72, block_size=512, window=2):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        self.window = window
        
        # 修改投影层：输出3*dim用于分割q/k/v
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, 2*dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, key_value):
        B, Lq, C = query.shape
        _, Lk, _ = key_value.shape
        
        # 动态填充保证可分块
        def pad_to_block(x, block_size):
            pad_len = (block_size - (x.size(1) % block_size)) % block_size
            return torch.nn.functional.pad(x, (0,0,0,pad_len)), pad_len
        
        # 对输入进行填充
        query_padded, q_pad = pad_to_block(query, self.block_size)
        key_padded, k_pad = pad_to_block(key_value, self.block_size)
        
        # 关键修正：正确分割q/k/v
        q = self.q_proj(query_padded)  # [B, L_padded, 3*C]
        k, v = self.kv_proj(key_padded).chunk(2, dim=-1)        
        # 分块重组
        q_blocks = rearrange(q, 'b (nq bsz) c -> b nq bsz c', bsz=self.block_size)  # [B, nq, bsz, C]
        k_blocks = rearrange(k, 'b (nk bsz) c -> b nk bsz c', bsz=self.block_size)
        v_blocks = rearrange(v, 'b (nk bsz) c -> b nk bsz c', bsz=self.block_size)
        
        # 生成块掩码
        num_q = q_blocks.size(1)
        num_k = k_blocks.size(1)
        block_indices = torch.arange(num_q, device=q.device)[:, None]
        allowed = (block_indices - torch.arange(num_k, device=q.device)).abs() <= self.window
        
        # 向量化注意力计算
        attn_scores = torch.einsum('bqsc,bkdc->bqksd', q_blocks, k_blocks) / (C**0.5)
        attn_scores = attn_scores.masked_fill(~allowed.unsqueeze(-1).unsqueeze(-1), -1e9)
        attn_weights = torch.softmax(attn_scores, dim=3)
        
        # 聚合结果
        output = torch.einsum('bqksd,bkdc->bqsc', attn_weights, v_blocks)
        output = rearrange(output, 'b q s c -> b (q s) c')[:, :Lq]  # [B, Lq, C]
        
        return self.out_proj(output)  # 输入输出维度均为C
    

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from einops import rearrange, repeat

class LinearCrossAttention(nn.Module):
    def __init__(self, 
                 dim=72, 
                 pos_dim=16,       # 位置编码维度
                 chunk_size=512,   # 分块大小（平衡显存和计算）
                 kernel_fn='elu',  # 核函数类型
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.eps = eps

        # 位置编码（假设输入是2D特征图展平）
        self.pos_enc = nn.Sequential(
            nn.Linear(2, pos_dim),   # 输入为(x,y)坐标
            nn.GELU(),
            nn.Linear(pos_dim, dim)
        )
        # 投影层（Q来自Transformer, KV来自CNN）
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        
        # 核函数选择
        if kernel_fn == 'elu':
            self.kernel = lambda x: F.elu(x) + 1.0  # 近似softmax的ELU+1核
        else:
            self.kernel = lambda x: torch.exp(x)     # 标准指数核

        # 输出归一化
        self.norm = nn.LayerNorm(dim)

    def forward(self, q, kv, coord_q, coord_kv):
        """
        输入:
            q: [batch, 16384, dim]   (Transformer输出)
            kv: [batch, 16384, dim] (CNN特征)
            coord_q: [16384, 2]     查询坐标 (归一化到[0,1])
            coord_kv: [16384, 2]     Key坐标
        """
        B, N, C = q.shape
        
        # ===== 1. 位置编码 =====
        pos_q = self.pos_enc(coord_q)   # [16384, dim]
        pos_k = self.pos_enc(coord_kv)  # [16384, dim]
        
        # 将位置编码融入Q/K
        q = self.to_q(q) + pos_q.unsqueeze(0)  # [B, N, C]
        k = self.to_k(kv) + pos_k.unsqueeze(0)
        v = self.to_v(kv)

        # 分块重组 (关键步骤)
        q_blocks = rearrange(q, 'b (n c) d -> b n c d', c=self.chunk_size)  # [B, 32, 512, C]
        k_blocks = rearrange(k, 'b (m c) d -> b m c d', c=self.chunk_size)
        v_blocks = rearrange(k, 'b (m c) d -> b m c d', c=self.chunk_size)
        
        # 批处理矩阵乘法 (一次性计算所有块间交互)
        attn = torch.einsum('bnic,bmjc->bnmij', q_blocks, k_blocks)  # [B,32,32,512,512]
        attn = attn.softmax(dim=-1)
        output = torch.einsum('bnmij,bmjc->bnic', attn, v_blocks)    # [B,32,512,C]
        
        return rearrange(output, 'b n c d -> b (n c d)')
        
        # # ===== 2. 分块计算 =====
        # q_chunks = q.split(self.chunk_size, dim=1)  # 分成多个块
        # k_chunks = k.split(self.chunk_size, dim=1)
        # v_chunks = v.split(self.chunk_size, dim=1)
        
        # outputs = []
        # for q_chunk in q_chunks:
        #     chunk_out = []
        #     for k_chunk, v_chunk in zip(k_chunks, v_chunks):
        #         # ===== 线性注意力核心计算 =====
        #         # 应用核函数
        #         K = self.kernel(k_chunk)  # [B, chunk, C]
        #         Q = self.kernel(q_chunk)  # [B, chunk, C]
                
        #         # 计算分母（稳定性处理）
        #         Z = 1.0 / (torch.einsum('bqc,bkc->bk', Q, K.sum(dim=1, keepdim=True)) + self.eps)
                
        #         # 分子部分
        #         V = torch.einsum('bkc,bvd->bcd', K, v_chunk)  # [B, C, C]
        #         out_chunk = torch.einsum('bqc,bcd,bk->bqd', Q, V, Z)  # [B, chunk, C]
                
        #         chunk_out.append(out_chunk)
            
        #     # 合并块结果
        #     outputs.append(torch.cat(chunk_out, dim=1))
        
        # # ===== 3. 合并与归一化 =====
        # output = torch.cat(outputs, dim=1)
        # return self.norm(output)

    @staticmethod
    def generate_coords(seq_len, img_size=128):
        """生成归一化坐标 [0,1]范围"""
        h = w = int(seq_len**0.5)
        assert h*w == seq_len, "序列长度需为平方数"
        
        y_coord = torch.linspace(0, 1, h)  # [h]
        x_coord = torch.linspace(0, 1, w)  # [w]
        grid_y, grid_x = torch.meshgrid(y_coord, x_coord)
        coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
        return coords  # [seq_len, 2]
    
# import torch
# import torch.nn as nn
import math
from torch.cuda.amp import autocast

def window_partition(x, window_size):
    """
    将输入张量按窗口划分
    Args:
        x: Tensor of shape (B, H, W, C)
        window_size (int): 窗口尺寸
    Returns:
        windows: Tensor of shape (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # 调整维度并合并窗口
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    将窗口结果还原为原始的特征图
    Args:
        windows: Tensor of shape (num_windows*B, window_size, window_size, C)
        window_size (int): 窗口尺寸
        H (int): 原特征图高度
        W (int): 原特征图宽度
    Returns:
        x: Tensor of shape (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SparseCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, stage=1):
        """
        稀疏交叉注意力模块
        Args:
            embed_dim (int): 嵌入维度
            num_heads (int): 注意力头数
            stage (int): 阶段编号（1,2,3,4），分别对应不同的token数量：
                         Stage1: N=16384 (128x128), Stage2: N=4096 (64x64),
                         Stage3: N=1024 (32x32), Stage4: N=256 (16x16)
        """
        super(SparseCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 用于生成q, k, v的线性投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 根据不同阶段选择合适的窗口尺寸
        if stage == 0:    # N=16384, 128x128 -> 使用窗口16x16
            self.window_size = 16
        elif stage == 1:  # N=4096, 64x64 -> 使用窗口8x8
            self.window_size = 8
        elif stage == 2:  # N=1024, 32x32 -> 使用窗口4x4
            self.window_size = 4
        elif stage == 3:  # N=256, 16x16 -> 使用窗口4x4
            self.window_size = 4
        else:
            raise ValueError("Unsupported stage. Stage must be one of 1,2,3,4.")
    
    def forward(self, q, kv):
        """
        前向计算
        Args:
            q: Tensor of shape (B, N, C)，来自Transformer block的输出（序列化后的结果）
            kv: Tensor of shape (B, N, C)，来自多次卷积后的特征图（序列化后的先验特征）
        Returns:
            out: Tensor of shape (B, N, C)
        """
        with autocast():
            B, N, C = q.shape
            # 假设q、kv序列对应的特征图为正方形
            H = W = int(math.sqrt(N))
            # 恢复成二维特征图
            q = q.view(B, H, W, C)
            kv = kv.view(B, H, W, C)
            
            # 线性变换得到查询、键和值
            q = self.q_proj(q)  # shape: (B, H, W, C)
            k = self.k_proj(kv)
            v = self.v_proj(kv)
            
            # 分头：变成 (B, H, W, num_heads, head_dim)
            q = q.view(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4).contiguous()
            k = k.view(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4).contiguous()
            v = v.view(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4).contiguous()
            
            # 合并 batch 与 head 维度便于后续窗口划分
            q = q.view(B * self.num_heads, H, W, self.head_dim)
            k = k.view(B * self.num_heads, H, W, self.head_dim)
            v = v.view(B * self.num_heads, H, W, self.head_dim)
            
            # 按窗口划分
            q_windows = window_partition(q, self.window_size)  # (num_windows*B*num_heads, window_size, window_size, head_dim)
            k_windows = window_partition(k, self.window_size)
            v_windows = window_partition(v, self.window_size)
            
            # 展平窗口内空间维度，形状变为 (num_windows*B*num_heads, window_area, head_dim)
            q_windows = q_windows.view(q_windows.shape[0], -1, self.head_dim)
            k_windows = k_windows.view(k_windows.shape[0], -1, self.head_dim)
            v_windows = v_windows.view(v_windows.shape[0], -1, self.head_dim)
            
            # 在每个窗口内计算缩放点积注意力
            attn = torch.matmul(q_windows, k_windows.transpose(-2, -1)) * self.scale
            with autocast(enabled=False):
                attn = torch.softmax(attn, dim=-1)
            out_windows = torch.matmul(attn, v_windows)  # (num_windows*B*num_heads, window_area, head_dim)
            
            # 恢复成窗口的二维形式
            out_windows = out_windows.view(-1, self.window_size, self.window_size, self.head_dim)
            # 将窗口合并还原成完整的特征图
            x = window_reverse(out_windows, self.window_size, H, W)  # (B*num_heads, H, W, head_dim)
            # 恢复成原始形状 (B, N, C)
            x = x.view(B, self.num_heads, H, W, self.head_dim).permute(0, 2, 3, 1, 4).contiguous().view(B, N, C)
            # 最后经过输出线性层
            x = self.out_proj(x)
        return x
    
# class MAttention(nn.Module):
#     def __init__(self, 
#                  dim=72, 
#                  pos_dim=16,       # 位置编码维度
#                  chunk_size=512,   # 分块大小（平衡显存和计算）
#                  kernel_fn='elu',  # 核函数类型
#                  eps=1e-6, 
#                  num_stage=0       # 当前阶段
#                  ):
#         super().__init__()
#         self.dim = dim
#         # self.chunk_size = chunk_size
#         # self.eps = eps
#         self.num_stage = num_stage
#         if self.num_stage == 1:
#             self.attn = LinearCrossAttention(dim=dim, pos_dim=pos_dim, chunk_size=chunk_size, kernel_fn=kernel_fn, eps=eps)
#         elif self.num_stage == 2:
#             self.attn = 

# 测试用例
if __name__ == "__main__":
    B, L, C = 4, 16384, 72
    q = torch.randn(B, L, C)
    kv = torch.randn(B, L, C)
    
    # 生成坐标（假设原始特征图大小128x128）
    coord = LinearCrossAttention.generate_coords(L, img_size=128)
    coord = coord.to(q.device)
    
    # 初始化模块
    attn = LinearCrossAttention(dim=C).to(q.device)
    
    # 前向计算
    output = attn(q, kv, coord, coord)
    print(output.shape)  # [4, 16384, 72]