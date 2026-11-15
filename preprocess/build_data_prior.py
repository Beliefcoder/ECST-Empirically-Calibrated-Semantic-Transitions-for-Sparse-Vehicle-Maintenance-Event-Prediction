import argparse
import os
import sys
from pathlib import Path
import numpy as np
from scipy import sparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_sparse, save_sparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/qwen25')
    parser.add_argument('--config', default='configs/salt_qwen25.json')
    args = parser.parse_args()
    import json
    config = json.loads(Path(args.config).read_text())
    train = Path(args.data_root) / 'standard/train'
    code_num = int(np.load(train / 'code_x.npz')['shape'][-1])
    matrix, _, _ = generate_decay_multi_transition_matrix(str(train), code_num,
        threshold_value=config['data_threshold'], topk=config['data_topk'],
        future_steps=config['future_steps'], decay=config['decay'])
    save_sparse(train / 'transition_T_decay_multi.npz', matrix.toarray())
    print('Saved empirical prior:', matrix.shape, matrix.nnz)

def _normalize_and_filter(matrix, row_base, threshold_value=0.01, topk=15):
    row_base = np.asarray(row_base, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    prob = np.zeros_like(matrix, dtype=np.float32)
    valid_rows = row_base > 0
    if np.any(valid_rows):
        prob[valid_rows] = matrix[valid_rows] / row_base[valid_rows, None]
    prob = prob / (prob + 1.0)
    np.fill_diagonal(prob, 0.0)
    filtered = np.zeros_like(prob, dtype=np.float32)
    for i in range(prob.shape[0]):
        row = prob[i]
        if topk is None or topk >= row.shape[0]:
            candidate_indices = np.where(row > threshold_value)[0]
        else:
            candidate_indices = np.argsort(row)[-topk:]
        for j in candidate_indices:
            if row[j] > threshold_value:
                filtered[i, j] = row[j]
    return sparse.csr_matrix(filtered)

def generate_decay_multi_transition_matrix(data_path, code_num, threshold_value=0.01, topk=100, future_steps=3, decay=0.7):
    packed = np.load(os.path.join(data_path, 'code_x.npz'))
    shape = tuple(packed['shape'])
    idx = packed['idx']
    code_x = sparse.csr_matrix((packed['values'], (idx[0] * shape[1] + idx[1], idx[2])), shape=(shape[0] * shape[1], shape[2]))
    visit_lens = np.load(os.path.join(data_path, 'visit_lens.npz'))['lens']
    weighted_counts = np.zeros((code_num, code_num), dtype=np.float32)
    code_freq = np.zeros(code_num, dtype=np.float32)
    for i in range(len(visit_lens)):
        visit_seq = code_x[i * shape[1]:(i + 1) * shape[1]].toarray()
        visit_len = int(visit_lens[i])
        for t in range(visit_len - 1):
            current_indices = np.where(visit_seq[t] > 0)[0]
            if len(current_indices) == 0:
                continue
            code_freq[current_indices] += 1.0
            max_future = min(visit_len, t + future_steps + 1)
            for future_t in range(t + 1, max_future):
                future_indices = np.where(visit_seq[future_t] > 0)[0]
                if len(future_indices) == 0:
                    continue
                weight = decay ** (future_t - t - 1)
                weighted_counts[np.ix_(current_indices, future_indices)] += weight
    T_sparse = _normalize_and_filter(weighted_counts, code_freq, threshold_value=threshold_value, topk=topk)
    return (T_sparse, weighted_counts, code_freq)


if __name__ == '__main__':
    main()
