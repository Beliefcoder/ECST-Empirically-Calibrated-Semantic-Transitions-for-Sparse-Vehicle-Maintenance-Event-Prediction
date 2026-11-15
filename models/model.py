import math
import torch
from torch import nn
from .layers import SampleAdaptiveFusionTransitionLayer, ImprovedTCRF, ResidualInteractionFusion

device = 'cuda'


class Model(nn.Module):
    def __init__(self, code_num, data_prior, semantic_prior, dropout=0.47):
        super().__init__()
        self.attention1 = SingleHeadAttentionLayer1(code_num, code_num, code_num, 50)
        self.classifier = Classifier(code_num, code_num, dropout)
        self.time_layer4 = nn.Linear(10, code_num)
        self.retain = RETAIN(inputDimSize=code_num, embDimSize=100, hiddenDimSize=10, numClass=code_num)
        self.selfattention = SelfAttention(10)
        self.tcrf = ImprovedTCRF(code_num=code_num)
        self.window_weights = nn.Parameter(torch.ones(3) / 3)
        self.lstm = nn.LSTM(code_num, 10, 10, batch_first=True)
        self.layernorm1 = nn.LayerNorm(10)
        self.layernorm2 = nn.LayerNorm(10)
        self.output1_norm = nn.LayerNorm(code_num)
        self.output2_norm = nn.LayerNorm(code_num)
        self.output1_scale = nn.Parameter(torch.tensor(1.0))
        self.output2_scale = nn.Parameter(torch.tensor(1.0))
        self.residual_interaction_fusion = ResidualInteractionFusion(code_num, max(128, code_num // 2), dropout)
        self.sample_adaptive_fusion_transition_layer = SampleAdaptiveFusionTransitionLayer(code_num, 10, dropout=dropout)
        self.sample_adaptive_fusion_transition_layer.set_transition_matrices(data_prior, semantic_prior)
        self.window_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, code_x, lengths):
        codes = self.retain(code_x, lengths)
        contexts = []
        for visits, length, context in zip(code_x, lengths, codes):
            repeated = context.unsqueeze(0).expand(int(length), -1)
            contexts.append(self.attention1(visits[:int(length)].float(), repeated, repeated).mean(0))
        context = self.output2_norm(torch.vstack(contexts)) * self.output2_scale
        visits = code_x.float()
        positions = torch.arange(visits.size(1), device=visits.device)[None, :]
        mask = positions < lengths[:, None]
        packed = nn.utils.rnn.pack_padded_sequence(visits, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.lstm(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=visits.size(1))
        encoded = self.layernorm1(encoded) * mask.unsqueeze(-1)
        encoded = self.selfattention(encoded, mask=mask)
        encoded = self.sample_adaptive_fusion_transition_layer(encoded, mask=mask)
        encoded = self.layernorm2(encoded) * mask.unsqueeze(-1)
        windows = []
        for window in [3, 10, None]:
            start = torch.zeros_like(lengths) if window is None else (lengths - window).clamp_min(0)
            window_mask = mask & (positions >= start[:, None])
            windows.append((encoded * window_mask.unsqueeze(-1)).sum(1))
        weights = self.window_weights.softmax(0)
        temporal = self.time_layer4(sum(weight * values for weight, values in zip(weights, windows)))
        temporal = self.output1_norm(temporal) * self.output1_scale
        fused, _ = self.tcrf(temporal, context, lengths)
        fused = self.residual_interaction_fusion(temporal, context, base_fused=fused)
        return self.classifier(fused)


class RETAIN(nn.Module):

    def __init__(self, inputDimSize=2813, embDimSize=100, hiddenDimSize=10, numClass=2813, dropout_rate=0.1):
        super(RETAIN, self).__init__()
        self.inputDimSize = inputDimSize
        self.relu = torch.nn.ReLU()
        self.inputDimSize = inputDimSize
        self.RNN = torch.nn.GRU(input_size=inputDimSize, hidden_size=hiddenDimSize, num_layers=5, batch_first=True, bidirectional=False)
        self.RNN1 = torch.nn.GRU(input_size=inputDimSize, hidden_size=hiddenDimSize, num_layers=5, batch_first=True, bidirectional=False)
        self.W_emb = nn.Parameter(torch.rand((inputDimSize, embDimSize)).float(), requires_grad=True)
        self.b_ema = nn.Parameter(torch.rand(embDimSize).float(), requires_grad=True)
        self.w_a = nn.Parameter(torch.rand(hiddenDimSize, 1).float(), requires_grad=True)
        self.b_a = nn.Parameter(torch.rand(1).float(), requires_grad=True)
        self.w_a1 = nn.Parameter(torch.rand(hiddenDimSize, 1).float(), requires_grad=True)
        self.b_a1 = nn.Parameter(torch.rand(1).float(), requires_grad=True)
        self.dropout_rate = dropout_rate
        self.W_output = nn.Parameter(torch.rand(embDimSize, numClass).float(), requires_grad=True)
        self.b_output = nn.Parameter(torch.zeros(numClass).float(), requires_grad=True)

    def forward(self, x, visit_lens):
        mask = torch.arange(x.shape[1], device=x.device)[None, :] < visit_lens[:, None]
        mask_attention = -1000000000.0 * (1 - mask.int()).unsqueeze(-1)
        x = x.float()
        self.W_emb = self.W_emb.float()
        self.b_ema = self.b_ema.float()
        x1 = self.relu(torch.matmul(x, self.W_emb) + self.b_ema)
        (x2, _) = self.RNN(x)
        a = nn.Softmax(dim=1)(torch.matmul(x2, self.w_a) + self.b_a + mask_attention)
        (x3, _) = self.RNN1(x)
        b = nn.Tanh()(torch.matmul(x3, self.w_a1) + self.b_a1)
        c = torch.sum(a * b * x, dim=1)
        return c

class Classifier(nn.Module):

    def __init__(self, input_size, output_size, dropout_rate=0.0):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.dropout(x)
        output = self.linear(output)
        return output

class SingleHeadAttentionLayer1(nn.Module):

    def __init__(self, query_size, key_size, value_size, attention_size):
        super().__init__()
        self.attention_size = attention_size
        self.dense_q = nn.Linear(query_size, attention_size)
        self.dense_k = nn.Linear(key_size, attention_size)
        self.dense_v = nn.Linear(value_size, value_size)

    def forward(self, q, k, v):
        query = self.dense_q(q.float())
        key = self.dense_k(k.float())
        value = self.dense_v(v.float())
        g = torch.div(torch.matmul(query, key.T), math.sqrt(self.attention_size))
        score = torch.softmax(g, dim=-1)
        output = torch.sum(torch.unsqueeze(score, dim=-1) * value, dim=-2)
        return output

class SelfAttention(nn.Module):

    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        (N, seq_length, embed_size) = x.shape
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)
        energy = torch.bmm(queries, keys.transpose(1, 2))
        energy = energy / embed_size ** (1 / 2)
        if mask is not None:
            key_mask = mask.unsqueeze(1)
            energy = energy.masked_fill(~key_mask, float('-inf'))
        attention = torch.softmax(energy, dim=2)
        attention = torch.nan_to_num(attention, nan=0.0)
        if mask is not None:
            query_mask = mask.unsqueeze(-1).to(attention.dtype)
            attention = attention * query_mask
        out = torch.bmm(attention, values)
        out = self.fc_out(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out
