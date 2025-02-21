import torch
from torch import nn
from torch.nn import functional as F
from mmcv.cnn import ConvModule
from timm.models.layers import DropPath, trunc_normal_

import math


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

        # Apply padding mask if present (attn_scores should be -inf where padding is)
        if input_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(input_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Softmax over the last dimension (key sequence length) to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (N, num_heads, Lq, Lin)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, values)  # (N, num_heads, Lq, head_dim)

        # Reshape back to (N, Lq, dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(N, Lq, self.dim)  # (N, Lq, dim)

        # Final output projection
        output = self.output_proj(attn_output)  # (N, Lq, dim)

        return output

    

class SparseAttention(nn.Module):
    def __init__(self, dim=256, num_heads=8, sparsity_factor=0.1):
        """
        Sparse Attention Module
        :param dim: hidden dimension  
        :param num_heads: number of attention heads
        :param sparsity_factor: proportion of keys to attend to
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError('dim must be divisible by num_heads, but got {} and {}'.format(dim, num_heads))
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sparsity_factor = sparsity_factor

        self.query_proj = nn.Linear(dim, dim)  # Projection for query
        self.key_proj = nn.Linear(dim, dim)    # Projection for keys (and values)
        self.output_proj = nn.Linear(dim, dim)  # Output projection

    def forward(self, query, input_flatten, input_padding_mask=None):
        """
        :param query: (N, Length_{query}, dim)
        :param input_flatten: (N, Len_in, dim)
        :param input_padding_mask: (N, Len_in), True for padding elements, False for non-padding elements 
        
        :return: output: (N, Length_{query}, dim)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        
        # Linear projections
        Q = self.query_proj(query)  # (N, Len_q, dim)
        K = self.key_proj(input_flatten)  # (N, Len_in, dim)
        
        # Reshape for multi-head attention
        Q = Q.view(N, Len_q, self.num_heads, self.head_dim).transpose(1, 2)  # (N, num_heads, Len_q, head_dim)
        K = K.view(N, Len_in, self.num_heads, self.head_dim).transpose(1, 2)  # (N, num_heads, Len_in, head_dim)

        # Compute attention scores
        attn_scores = torch.einsum('nhqd,nhkd->nhqk', Q, K)  # (N, num_heads, Len_q, Len_in)
        
        # Apply padding mask
        if input_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(input_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Apply sparsity
        num_keys_to_select = int(self.sparsity_factor * Len_in)
        topk_scores, topk_indices = torch.topk(attn_scores, num_keys_to_select, dim=-1)

        # Create a sparse attention matrix
        sparse_attn_weights = F.softmax(topk_scores, dim=-1)
        sparse_attn_weights = sparse_attn_weights / sparse_attn_weights.sum(dim=-1, keepdim=True)  # Normalize

        # Gather values based on top-k indices
        V = input_flatten.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # (N, num_heads, Len_in, dim) 
        selected_V = torch.gather(V, 2, topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.dim))  # (N, num_heads, Len_q, dim)

        # Compute output
        output = torch.einsum('nhql,nhld->nhqd', sparse_attn_weights, selected_V)  # (N, num_heads, Len_q, dim)

        output = output.transpose(1, 2).contiguous().view(N, Len_q, self.dim)  # (N, Len_q, dim)
        output = self.output_proj(output)  # (N, Len_q, dim)

        return output