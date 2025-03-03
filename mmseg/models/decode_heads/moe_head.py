# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from torch import Tensor
from timm.models.layers import trunc_normal_
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        # [B, H*W, C]
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        # [B, H*W, C]
        return x

class MixFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

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

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SwitchGate(nn.Module):
    """
    SwitchGate module for MoE (Mixture of Experts) model.

    Args:
        dim (int): Input dimension.
        num_experts (int): Number of experts.
        capacity_factor (float, optional): Capacity factor for sparsity. Defaults to 1.0.
    """

    def __init__(
        self,
        dim,
        num_experts: int,
        capacity_factor: float = 1.0,
        epsilon: float = 1e-6
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x: Tensor, use_aux_loss=False):
        """
        Forward pass of the SwitchGate module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Gate scores.
        """
        # Compute gate scores
        gate_scores = F.softmax(self.w_gate(x), dim=-1)

        # Determine the top-1 expert for each token
        capacity = int(self.capacity_factor * x.size(0))

        top_k_scores, top_k_indices = gate_scores.topk(1, dim=-1)

        # Mask to enforce sparsity
        mask = torch.zeros_like(gate_scores).scatter_(
            1, top_k_indices, 1
        )

        # Combine gating scores with the mask
        masked_gate_scores = gate_scores * mask

        # Denominators
        denominators = (
            masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        )

        # Norm gate scores to sum to the capacity
        gate_scores = (masked_gate_scores / denominators) * capacity

        if use_aux_loss:
            load = gate_scores.sum(0)  # Sum over all examples
            importance = gate_scores.sum(1)  # Sum over all experts

            # Aux loss is mean suqared difference between load and importance
            loss = ((load - importance) ** 2).mean()

            return gate_scores, loss

        return gate_scores, None

class SwitchMoE(nn.Module):
    """
    A module that implements the Switched Mixture of Experts (MoE) architecture.

    Args:
        dim (int): The input dimension.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float, optional): The capacity factor that controls the capacity of the MoE. Defaults to 1.0.
        mult (int, optional): The multiplier for the hidden dimension of the feedforward network. Defaults to 4.

    Attributes:
        dim (int): The input dimension.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float): The capacity factor that controls the capacity of the MoE.
        mult (int): The multiplier for the hidden dimension of the feedforward network.
        experts (nn.ModuleList): The list of feedforward networks representing the experts.
        gate (SwitchGate): The switch gate module.

    """

    def __init__(
        self,
        dim: int,
        output_dim: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        mult: int = 4,
        use_aux_loss: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.mult = mult
        self.use_aux_loss = use_aux_loss

        self.experts = nn.ModuleList(
            [
                # FeedForward(dim, dim, mult, *args, **kwargs)
                MixFFN(dim, mult * dim, output_dim)
                for _ in range(num_experts)
            ]
        )

        self.gate = SwitchGate(
            dim,
            num_experts,
            capacity_factor,
        )

    def forward(self, x: Tensor, H, W):
        """
        Forward pass of the SwitchMoE module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor of the MoE.

        """
        # (batch_size, seq_len, num_experts)
        gate_scores, loss = self.gate(
            x, use_aux_loss=self.use_aux_loss
        )

        # Dispatch to experts
        # expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = [expert(x, H, W) for expert in self.experts]
        # Check if any gate scores are nan and handle
        if torch.isnan(gate_scores).any():
            print("NaN in gate scores")
            gate_scores[torch.isnan(gate_scores)] = 0

        # Stack and weight outputs
        stacked_expert_outputs = torch.stack(
            expert_outputs, dim=-1
        )  # (batch_size, seq_len, output_dim, num_experts)
        if torch.isnan(stacked_expert_outputs).any():
            stacked_expert_outputs[
                torch.isnan(stacked_expert_outputs)
            ] = 0

        # Combine expert outputs and gating scores
        moe_output = torch.sum(
            gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1
        )

        return moe_output, loss
    
@HEADS.register_module()
class MoEHead(BaseDecodeHead):
    """
    # MoEHead
    """
    def __init__(self, decoder_params=None, **kwargs):
        super(MoEHead, self).__init__(input_transform='multiple_select', **kwargs)
        # assert len(feature_strides) == len(self.in_channels)
        # assert min(feature_strides) == feature_strides[0]
        # self.feature_strides = feature_strides

        # 四个输入维度
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = decoder_params
        # 输出维度
        embed_dim = decoder_params['embed_dim']
        # 专家网络（每个特征图对应一个MoE）
        self.ffn_layers = nn.ModuleList([
            SwitchMoE(dim=channel, output_dim=embed_dim, num_experts=2) for channel in self.in_channels
        ])
        # self.norm_layers = nn.ModuleList([
        #     nn.LayerNorm(channel) for channel in self.in_channels
        # ])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv = nn.ModuleList([
            nn.Conv2d(c, embed_dim, kernel_size=1) for c in self.in_channels
        ])
        self.linear_fuse = ConvModule(
            in_channels=embed_dim*4,
            out_channels=embed_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.linear_pred = nn.Conv2d(embed_dim, self.num_classes, kernel_size=1)
        # 初始化权重
        self.apply(self._init_weights)

        # 检查参数是否已初始化
        # for name, param in self.named_parameters():
        #     print(f"{name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")

    def _init_weights(self, m):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 对卷积层使用 Kaiming 初始化（He 初始化）
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 对线性层使用 Xavier 初始化
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
                # 对归一化层初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):

        
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        B, _, H1, W1 = c1.shape  # 获取 c1 的大小
        
        feature_maps = []
        for i, c in enumerate([c1, c2, c3, c4]):
            # 上采样到 c1 的大小
            c = F.interpolate(c, size=c1.size()[2:], mode='bilinear', align_corners=False) # [B, c, H1, W1]
            _c = self.conv[i](c) # [B, embed_dim, H1, W1]
            _c = _c.view(_c.size(0), _c.size(1), -1).transpose(1,2).contiguous() # [B, H1*W1, embed_dim]
            
            # 通过对应的 MoE
            x, _ = self.ffn_layers[i](c.view(c.size(0), c.size(1), -1).transpose(1,2).contiguous(), H1, W1)  # [B, H1*W1, embed_dim]
            x = _c + x
            # x = self.norm_layers[i](x).transpose(1, 2).view_as(c)
            x = self.norm(x).transpose(1,2).contiguous().view(B, -1, H1, W1)
            feature_maps.append(x)

        # 特征图拼接，融合 
        fused_features = torch.cat(feature_maps, dim=1) # (B, 4*embed_dim, H1, W1)
        x = self.linear_fuse(fused_features) # (B, embed_dim, H1, W1)
        x = self.linear_pred(x) #[B, num_classes, H1, W1]
        
        return x
