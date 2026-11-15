import math
import numpy as np
import torch
from torch import nn
from scipy import sparse


def _coerce_transition_tensor(value, device, dtype=torch.float32):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    array = value.toarray() if sparse.issparse(value) else np.array(value)
    return torch.as_tensor(array, device=device, dtype=dtype)

class SampleAdaptiveFusionTransitionLayer(nn.Module):

    def __init__(self, code_num, hidden_size, init_alpha=0.5, init_beta=0.5, dropout=0.1, gate_mode='sample_adaptive', shared_projection=False):
        super().__init__()
        self.code_num = code_num
        self.hidden_size = hidden_size
        self.gate_mode = gate_mode
        self.shared_projection = shared_projection
        self.linear_data = nn.Linear(hidden_size, code_num)
        self.linear_llm = nn.Linear(hidden_size, code_num)
        self.output_linear_data = nn.Linear(code_num, hidden_size)
        self.output_linear_llm = nn.Linear(code_num, hidden_size)
        for layer in [self.linear_data, self.linear_llm, self.output_linear_data, self.output_linear_llm]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.alpha_mlp = nn.Sequential(nn.Linear(hidden_size, max(8, hidden_size)), nn.ReLU(), nn.Dropout(dropout), nn.Linear(max(8, hidden_size), 1))
        self.alpha_bias = nn.Parameter(torch.tensor(float(init_alpha)))
        init_alpha_clamped = min(max(float(init_alpha), 0.0001), 1.0 - 0.0001)
        self.global_alpha_logit = nn.Parameter(torch.tensor(math.log(init_alpha_clamped / (1.0 - init_alpha_clamped))))
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))
        self.dropout = nn.Dropout(dropout)
        self._T_data_tensor = None
        self._T_llm_tensor = None

    def set_transition_matrices(self, T_data, T_llm):
        if T_data is not None:
            self._T_data_tensor = torch.from_numpy(T_data.toarray() if sparse.issparse(T_data) else np.array(T_data)).float()
        if T_llm is not None:
            self._T_llm_tensor = torch.from_numpy(T_llm.toarray() if sparse.issparse(T_llm) else np.array(T_llm)).float()

    def _select_last_valid(self, h, mask=None):
        if mask is None:
            return h[:, -1, :]
        lengths = mask.long().sum(dim=1).clamp(min=1)
        batch_idx = torch.arange(h.size(0), device=h.device)
        return h[batch_idx, lengths - 1]

    def forward(self, h, mask=None, transition_T=None, transition_T_llm=None, return_extras=False):
        T_data_tensor = self._T_data_tensor if transition_T is None else _coerce_transition_tensor(transition_T, device=h.device)
        T_llm_tensor = self._T_llm_tensor if transition_T_llm is None else _coerce_transition_tensor(transition_T_llm, device=h.device)
        if T_data_tensor is None or T_llm_tensor is None:
            return (h, None) if return_extras else h
        T_data_tensor = T_data_tensor.to(h.device)
        T_llm_tensor = T_llm_tensor.to(h.device)
        z_data = self.output_linear_data(torch.matmul(self.linear_data(h), T_data_tensor))
        if self.shared_projection:
            z_llm = self.output_linear_data(torch.matmul(self.linear_data(h), T_llm_tensor))
        else:
            z_llm = self.output_linear_llm(torch.matmul(self.linear_llm(h), T_llm_tensor))
        if self.gate_mode == 'fixed':
            alpha = h.new_full((h.size(0), 1, 1), 0.5)
        elif self.gate_mode == 'global':
            alpha = torch.sigmoid(self.global_alpha_logit).view(1, 1, 1).expand(h.size(0), 1, 1)
        else:
            last_context = self._select_last_valid(h, mask)
            alpha = torch.sigmoid(self.alpha_mlp(last_context) + self.alpha_bias).view(-1, 1, 1)
        beta = torch.sigmoid(self.beta)
        fused = alpha * z_data + (1.0 - alpha) * z_llm
        fused = self.dropout(fused)
        output = h + beta * fused
        if mask is not None:
            output = output * mask.unsqueeze(-1).to(output.dtype)
        extras = {'fusion_alpha_mean': alpha.mean().item(), 'fusion_alpha_std': alpha.std().item(), 'fusion_alpha_values': alpha.view(-1).detach().cpu().numpy(), 'residual_beta': beta.item(), 'gate_mode': self.gate_mode, 'shared_projection': self.shared_projection}
        if transition_T_llm is not None and isinstance(transition_T_llm, torch.Tensor):
            extras['end_to_end_transition_mean'] = transition_T_llm.mean().item()
            extras['end_to_end_transition_density'] = (transition_T_llm > 0).float().mean().item()
        return (output, extras) if return_extras else output

def compute_simple_fusion(x, y, mode='sum_mul'):
    if mode == 'mul':
        return x * y
    if mode == 'sum_mul':
        return x + y + x * y
    raise ValueError(f'Unsupported simple fusion mode: {mode}')

class ResidualInteractionFusion(nn.Module):

    def __init__(self, code_num, hidden_dim=512, dropout=0.1, simple_fusion_mode='sum_mul'):
        super().__init__()
        self.simple_fusion_mode = simple_fusion_mode
        self.residual_mlp = nn.Sequential(nn.Linear(code_num * 4, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, code_num))
        self.output_norm = nn.LayerNorm(code_num)
        self.residual_scale = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x, y, base_fused=None, return_extras=False):
        interaction = torch.cat([x, y, x - y, x * y], dim=-1)
        residual = self.residual_mlp(interaction)
        residual = self.output_norm(residual)
        alpha = torch.sigmoid(self.residual_scale)
        base = base_fused if base_fused is not None else compute_simple_fusion(x, y, mode=self.simple_fusion_mode)
        output = base + alpha * residual
        extras = None
        if return_extras:
            extras = {'interaction_alpha': alpha.item(), 'interaction_residual_norm': residual.norm(dim=-1).mean().item()}
        return (output, extras) if return_extras else output

class ImprovedTCRF(nn.Module):

    def __init__(self, code_num, embed_dim=256, hidden_dim=512, num_experts=3, dropout=0.3, simple_fusion_mode='sum_mul'):
        super().__init__()
        self.code_num = code_num
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.simple_fusion_mode = simple_fusion_mode
        self.input_proj = nn.Sequential(nn.Linear(code_num, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.time_encoder = nn.Sequential(nn.Linear(1, embed_dim), nn.LayerNorm(embed_dim), nn.Sigmoid())
        router_input_dim = embed_dim * 2 + embed_dim
        self.router = nn.Sequential(nn.Linear(router_input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_experts))
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, embed_dim)) for _ in range(num_experts)])
        self.output_proj = nn.Linear(embed_dim, code_num)
        self.simple_fusion_weight = nn.Parameter(torch.tensor(0.7))
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, y, seq_len=None, return_extras=False):
        B = x.shape[0]
        device = x.device
        simple_fusion = compute_simple_fusion(x, y, mode=self.simple_fusion_mode)
        x_feat = self.input_proj(x)
        y_feat = self.input_proj(y)
        if seq_len is not None:
            if seq_len.dim() == 0:
                seq_len = seq_len.unsqueeze(0).expand(B)
            time_emb = self.time_encoder(seq_len.float().unsqueeze(-1))
        else:
            time_emb = torch.zeros(B, self.embed_dim, device=device)
        router_input = torch.cat([x_feat, y_feat, time_emb], dim=-1)
        expert_weights = torch.softmax(self.router(router_input), dim=-1)
        concat_feat = torch.cat([x_feat, y_feat], dim=-1)
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(concat_feat))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        tcrf_fused = torch.sum(expert_outputs * expert_weights.unsqueeze(-1), dim=1)
        tcrf_fused = self.layer_norm(tcrf_fused)
        tcrf_fused = self.dropout(tcrf_fused)
        tcrf_output = self.output_proj(tcrf_fused)
        alpha = torch.sigmoid(self.simple_fusion_weight)
        final_output = alpha * simple_fusion + (1 - alpha) * tcrf_output
        extras = None
        if return_extras:
            extras = {'expert_weights': expert_weights.detach().cpu().numpy(), 'simple_fusion_weight': alpha.item(), 'simple_fusion_mode': self.simple_fusion_mode}
        return (final_output, extras)

    def get_fusion_weight(self):
        return torch.sigmoid(self.simple_fusion_weight).item()
