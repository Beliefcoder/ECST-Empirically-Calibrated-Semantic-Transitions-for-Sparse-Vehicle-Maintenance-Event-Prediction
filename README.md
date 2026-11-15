# SALT

## Installation

Run from the repository root:

```bash
conda create -n salt python=3.10 -y
conda activate salt
pip install -r requirements.txt
```



## Model Setup

Download the **Qwen/Qwen2.5-7B base model** (not Instruct) and place all model files in `models/Qwen2.5-7B/`, including the configuration, tokenizer, weight shards, and index files. Alternatively, specify another location with `--model-path`.

## Running

Place `ADMISSIONS.csv`, `DIAGNOSES_ICD.csv`, and `code_description_map.json` in `data/raw/`, then run from the repository root:

```bash
python preprocess/build_dataset.py
python preprocess/build_data_prior.py
python preprocess/build_semantic_prior_qwen25.py --model-path models/Qwen2.5-7B --device cuda:0
python train.py --device cuda:0 --output outputs/salt_seed50
```

Please contact the authors to obtain the data.
