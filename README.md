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

## Data Format and Field Mapping

The preprocessing interface retains legacy healthcare-style filenames and column names for compatibility with the reused sequence-processing code. In the vehicle-maintenance task, these fields are interpreted as follows; their names do not denote hospital admissions or medical diagnoses.

| File | Field | Meaning in this task |
|---|---|---|
| `ADMISSIONS.csv` | `SUBJECT_ID` | De-identified vehicle identifier |
| `ADMISSIONS.csv` | `HADM_ID` | Maintenance-visit identifier |
| `ADMISSIONS.csv` | `ADMITTIME` | Maintenance-visit timestamp used for chronological ordering |
| `DIAGNOSES_ICD.csv` | `HADM_ID` | Maintenance-visit identifier linking an event to its visit |
| `DIAGNOSES_ICD.csv` | `ICD9_CODE` | Maintenance-event label code; it is not interpreted as an ICD-9 diagnosis code |
| `code_description_map.json` | `code_to_part` | Mapping from maintenance-event codes to component or event names |

The loader groups visits by vehicle, orders them by timestamp, and represents each visit as a multi-hot maintenance-event vector. Vehicles with at least five eligible visits are retained. The final visit supplies the prediction target; preceding visits supply the historical input. Splitting is performed at vehicle level.

Examples from the supplied vehicle-label metadata are shown below. English translations are provided for readability; the Chinese names are the metadata values used to construct semantic-encoder inputs.

| Label code | Metadata name | English translation |
|---|---|---|
| `C0001` | 机油滤清器 | Engine oil filter |
| `C0003` | 机油 | Engine oil |
| `C0103` | 喷油器 | Fuel injector |
| `C0209` | 涡轮增压器 | Turbocharger |
| `C0241` | 发动机水泵 | Engine water pump |

These examples document the label semantics and interface mapping; they do not independently establish the provenance of the underlying records. Dataset provenance and access should be confirmed with the authors.

Please contact the authors to obtain the data.
