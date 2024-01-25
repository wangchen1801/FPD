import sys
from torch import nn
from mmfewshot.detection.models.utils.aggregation_layer import AGGREGATORS
from mmcv.runner import BaseModule

import math
import torch
import torch.nn.functional as F


@AGGREGATORS.register_module()
class PrototypesDistillation(BaseModule):
    def __init__(self, num_queries, dim, num_base_cls=15, num_novel=0):
        super().__init__()
        self.num_queries = num_queries
        self.num_base_cls = num_base_cls
        self.num_novel = num_novel
        self.dim = dim

        k_dim = dim // 4
        self.k_dim = k_dim

        self.query_embed = nn.Embedding(num_queries*num_base_cls, k_dim)
        self.duplicated = False
        if self.num_novel > 0:
            self.query_embed_novel = nn.Embedding(num_queries * num_novel, k_dim)
        self.w_qk = nn.Linear(dim, k_dim, bias=False)

    def forward(self, support_feats, support_gt_labels=None, forward_novel=False, forward_novel_test=False):
        """
        Args:
            support_feats: Tensor with shape (B, C, H, W).
            support_gt_labels: Support gt labels.
            forward_novel (bool): Novel classes.
            forward_novel_test (bool): Test time.
        Returns:
            tensor with shape (15, 1024)
        """

        # at the fine-tuning stage, duplicate the most compatible feature queries for the novel classes
        # ************************************************************
        if not self.duplicated and self.num_novel > 0 and not forward_novel_test and forward_novel:
            with torch.no_grad():
                support_feats_mp = F.max_pool2d(support_feats, kernel_size=2, stride=2)
                B, C, H, W = support_feats_mp.shape
                k = support_feats_mp.reshape(B, C, H*W).permute(0, 2, 1)  # (B, 196, 1024)
                k = self.w_qk(k)  # (B, 196, 1024)
                query_emb = self.query_embed.weight
                q = query_emb.unsqueeze(0).repeat(B, 1, 1)
                B, Nt, E = q.shape
                attn = torch.bmm(q / math.sqrt(E), k.transpose(-2, -1))
                weight = torch.topk(attn, 20, dim=-1)[0].mean(-1)

                drop = 5
                top_indices = torch.topk(weight, self.num_queries + drop, dim=-1)[1][:, -self.num_queries:]
                top_emb = torch.gather(self.query_embed.weight.unsqueeze(0).expand(B, -1, -1), 1, top_indices.unsqueeze(-1).expand(-1, -1, self.k_dim))
                top_emb = top_emb[torch.sort(support_gt_labels, dim=0)[1]].reshape(self.num_novel*self.num_queries, self.k_dim)
                self.query_embed_novel.weight.copy_(top_emb)
            self.duplicated = True  # set as True once duplicated
        # ************************************************************

        support_feats_mp = F.max_pool2d(support_feats, kernel_size=2, stride=2)
        B, C, H, W = support_feats_mp.shape
        k = v = support_feats_mp.reshape(B, C, H*W).permute(0, 2, 1)  # (B, 196, 1024)
        k = self.w_qk(k)  # (B, 196, 1024)

        # scaled dot-product attention
        if forward_novel:
            query_emb = self.query_embed_novel.weight
            q = query_emb.reshape(self.num_novel, self.num_queries, query_emb.size(-1))
        else:
            query_emb = self.query_embed.weight
            q = query_emb.reshape(self.num_base_cls, self.num_queries, query_emb.size(-1))
        q = q[support_gt_labels, ...]  # align with support_gt_labels
        B, Nt, E = q.shape
        attn = torch.bmm(q / math.sqrt(E), k.transpose(-2, -1))
        weight = torch.topk(attn, 20, dim=-1)[0].mean(-1)
        prototypes = torch.matmul(attn.softmax(-1), v)     # (B, 5, 1024)

        return weight, prototypes


@AGGREGATORS.register_module()
class PrototypesAssignment(BaseModule):
    def __init__(self, dim, num_bg=5):
        super().__init__()
        k_dim = dim // 4
        self.w_qk = nn.Linear(dim, k_dim, bias=False)

        self.num_bg = num_bg
        if self.num_bg > 0:
            self.dummy = nn.Parameter(torch.Tensor(self.num_bg, dim))
            nn.init.normal_(self.dummy)
            self.linear = nn.Linear(dim, k_dim)
        self.gamma = nn.Parameter(torch.tensor(0.))

    def forward(self, query_feature, prototypes, query_img_metas=None):
        """
        Args:
            query_feature: Tensor with shape (B, C, H, W)
            prototypes: Tensor with shape (num_supp, num_queries, C),
            query_img_metas: Visualization.
        Returns:
            class-specific query feature: tensor(B, C, H, W)
        """

        B, C, H, W = query_feature.shape
        num_supp, num_queries, _ = prototypes.shape
        q = query_feature.reshape(B, C, H*W).permute(0, 2, 1)   # (B, H*W, 1024)
        k = v = prototypes.reshape(num_supp * num_queries, C)

        q = self.w_qk(q)
        k = self.w_qk(k)

        if self.num_bg > 0:
            dummy_v = torch.zeros((self.num_bg, C), device='cuda')
            k = torch.cat([k, self.linear(self.dummy)], dim=0)
            v = torch.cat([v, dummy_v], dim=0)

        k = k.unsqueeze(0)
        B, Nt, E = q.shape
        attn = torch.bmm(q / math.sqrt(E), k.expand(B, -1, -1).transpose(-2, -1))
        attn.div_(0.5)

        out = torch.matmul(attn.softmax(-1), v)  # (B, 2850, 1024)
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)
        out = query_feature + self.gamma * out
        return out

