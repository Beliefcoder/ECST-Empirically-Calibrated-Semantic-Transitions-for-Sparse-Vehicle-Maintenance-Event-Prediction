import argparse
import json
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import save_sparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default='data/raw')
    parser.add_argument('--output', default='data/qwen25')
    parser.add_argument('--split-seed', type=int, default=6669)
    parser.add_argument('--descriptions', default=None)
    args = parser.parse_args()
    raw, root = Path(args.raw), Path(args.output)
    metadata_path = Path(args.descriptions) if args.descriptions else raw / 'code_description_map.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    admissions = pd.read_csv(raw / 'ADMISSIONS.csv')
    diagnoses = pd.read_csv(raw / 'DIAGNOSES_ICD.csv', dtype={'ICD9_CODE': str})
    codes = {admission: group['ICD9_CODE'].drop_duplicates().tolist()
             for admission, group in diagnoses.groupby('HADM_ID', sort=False)}
    admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'], errors='raise')
    patients = OrderedDict()
    for patient, group in admissions.groupby('SUBJECT_ID', sort=False):
        visits = group.sort_values('ADMITTIME', kind='stable')['HADM_ID'].tolist()
        visits = [visit for visit in visits if visit in codes]
        if len(visits) >= 5:
            patients[patient] = visits
    mapping = OrderedDict()
    for visits in patients.values():
        for visit in visits:
            for code in codes[visit]:
                if code not in mapping:
                    mapping[code] = len(mapping)
    codes = {admission: [mapping[code] for code in values if code in mapping]
             for admission, values in codes.items()}
    identifiers = list(patients)
    np.random.RandomState(args.split_seed).shuffle(identifiers)
    train_count, valid_count = int(len(identifiers) * .8), int(len(identifiers) * .1)
    splits = {'train': identifiers[:train_count],
              'valid': identifiers[train_count:train_count + valid_count],
              'test': identifiers[train_count + valid_count:]}
    if not patients or any(not values for values in splits.values()):
        raise ValueError('At least one eligible vehicle is required in each split')
    maximum = max(map(len, patients.values()))
    root.mkdir(parents=True, exist_ok=False)
    (root / 'encoded').mkdir()
    with (root / 'encoded/code_map.pkl').open('wb') as handle:
        pickle.dump(mapping, handle)
    with (root / 'encoded/pids.pkl').open('wb') as handle:
        pickle.dump({name + '_pids': values for name, values in splits.items()}, handle)
    (root / 'code_description_map.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    for name, selected in splits.items():
        path = root / 'standard' / name
        path.mkdir(parents=True)
        visits = np.zeros((len(selected), maximum, len(mapping)), dtype=np.uint8)
        labels = np.zeros((len(selected), len(mapping)), dtype=np.uint8)
        lengths = np.zeros(len(selected), dtype=np.int64)
        for index, patient in enumerate(selected):
            history = patients[patient]
            lengths[index] = len(history) - 1
            for position, admission in enumerate(history[:-1]):
                visits[index, position, codes[admission]] = 1
            labels[index, codes[history[-1]]] = 1
        save_sparse(path / 'code_x.npz', visits)
        save_sparse(path / 'code_y.npz', labels)
        np.savez_compressed(path / 'visit_lens.npz', lens=lengths)
        print(name, visits.shape, int(labels.sum()), flush=True)


if __name__ == '__main__':
    main()