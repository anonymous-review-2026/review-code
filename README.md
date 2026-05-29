# CGTFNet-anon

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data layout

This repository does not include any dataset files or split files.

Place serialized datasets under:

- `Dataset/Data/dataset_adni116_AAL116.save`
- `Dataset/Data/dataset_abide116_AAL116.save`

Precomputed fold split files are loaded/created under:

- `Dataset/Splits/adni116_5fold_seed42.npz`
- `Dataset/Splits/abide116_5fold_seed42.npz`
