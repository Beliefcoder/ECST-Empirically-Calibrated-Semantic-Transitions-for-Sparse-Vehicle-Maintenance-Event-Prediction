# Raw Data

The local release preparation copy contains the main experiment's raw inputs:

- `ADMISSIONS.csv`: `SUBJECT_ID`, `HADM_ID`, `ADMITTIME`.
- `DIAGNOSES_ICD.csv`: `SUBJECT_ID`, `HADM_ID`, `ICD9_CODE`.
- `code_description_map.json`: the `code_to_part` mapping used for label prompts.

These legacy field names identify vehicles, maintenance visits, dates, and maintenance codes in this task. Visits are sorted by date. Eligible vehicles have at least five visits; the final visit is the target. The fixed split seed is 6669, with 80/10/10 percent train/validation/test partitions. Label indices follow first appearance while traversing vehicles in input order and their visits in chronological order.

Distribution permission and de-identification must be confirmed before publishing the real records. Copying them into this local directory does not grant permission to redistribute them.