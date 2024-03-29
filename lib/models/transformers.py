import math

import torch
import torch.nn as nn


class MFIE(nn.Module):

    def __init__(self, bs, hidden_size, head_num):
        super(MFIE, self).__init__()
        self.hv = SelfAttention(hidden_size, head_num)
        self.w = nn.Parameter(torch.randn(bs, 1, hidden_size))

    def forward(self, feature1, feature2):
        x = self.hv(feature1, feature2)
        embedding_shift = x + feature1
        # embedding_shift = x + torch.mul(self.w, feature1)
        # embedding_shift = x
        return embedding_shift


# class CFIE_A_I(nn.Module):
#
#     def __init__(self, hidden_size, image_size, audio_size):
#         super(CFIE_A_I, self).__init__()
#         self.hv = SelfAttention(hidden_size)
#         self.conv1d = nn.Conv1d(image_size, audio_size, kernel_size=1)
#
#     def forward(self, audio_embedding, visual=None, ):
#         visual_ = self.hv(audio_embedding, visual)
#
#         visual_ = self.conv1d(visual_)
#
#         embedding_shift = visual_ + visual
#
#         return embedding_shift


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, head_num=1):
        super(SelfAttention, self).__init__()
        self.head_num = head_num
        self.s_d = hidden_size // self.head_num
        self.all_head_size = self.head_num * self.s_d
        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x):
        x = x.view(x.size(0), x.size(1), self.head_num, -1)
        return x.permute(0, 2, 1, 3)

    def forward(self, wave_embedding, embedding):
        Q = self.Wq(wave_embedding)
        K = self.Wk(embedding)
        V = self.Wv(embedding)
        Q = self.transpose_for_scores(Q)
        K = self.transpose_for_scores(K)
        V = self.transpose_for_scores(V)
        weight_score = torch.matmul(Q, K.transpose(-1, -2))
        weight_prob = nn.Softmax(dim=-1)(weight_score / (self.head_num ** 0.5))
        # weight_prob = nn.Softmax(dim=-1)(weight_score * 8)

        cross_layer = torch.matmul(weight_prob, V)
        cross_layer = cross_layer.permute(0, 2, 1, 3).contiguous()
        new_cross_layer_shape = cross_layer.size()[:-2] + (self.all_head_size,)
        cross_layer = cross_layer.view(*new_cross_layer_shape)

        return cross_layer
