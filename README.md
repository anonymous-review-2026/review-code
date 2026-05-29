# CGTFNet-anon

This repository keeps only the CGTFNet method components extracted from the original training path
for SPD-fusion training on ADNI116/ABIDE116.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train/evaluate (5-fold):

```bash
python tester.py -d adni116 -m cgtfnet --device 0 --name adni116_cgtfnet_5fold --epochs 20 --folds 5 --batch_size 16 --use_spd_fusion true --debug_model True --window_size 20 --shift_size 8 --fringe_size 0 --fringe_coeff 0 --n_layers 1 --dynamic_length none --crt_dropout 0 --crt_attnum 8 --crt_attn_dim 16 --crt_window_attn true --debug_step_logs true --debug_every 1000 --label_smoothing 0 --crt_weight_decay 0 --crt_detach_spd_q true --num_heads 32 --head_dim 20 --seed 41 --lr_scheduler none
```

## Data layout

This repository does not include any dataset files or split files.

Place serialized datasets under:

- `Dataset/Data/dataset_adni116_AAL116.save`
- `Dataset/Data/dataset_abide116_AAL116.save`

Precomputed fold split files are loaded/created under:

- `Dataset/Splits/adni116_5fold_seed42.npz`
- `Dataset/Splits/abide116_5fold_seed42.npz`
