# Mol-Mamba для предсказания радиационной стабильности (dose constant)

Пайплайн для регрессии **dose constant** состоит из:
- атомного графа молекулы (узлы — атомы, рёбра — связи) → **Mol-Mamba encoder** (GNN → упорядочение → Mamba/GRU);
- фрагментного графа (кольцевые системы) → **Frag GNN encoder**;
- числовых признаков молекулы и растворителя (раздельные векторы, включая молекулярные дескрипторы и параметры среды);
- **Mamba-Transformer fuser** для слияния `[ATOM | FRAG | NUM_mol | NUM_solvent]`;
- двухступенчатого SSL-предобучения (SDA + e-semantic masked fusion) на молекуле (граф и молекулярные дескрипторы);
- файнтюнинга с **scaffold k-fold CV**.

## Дерево проекта
```
molmamba_radstab/
├── data.py
├── models/
│   ├── mol_mamba.py
│   ├── frag_mamba.py
│   ├── fuser.py
│   └── predictor.py
├── ssl_pretrain.py
├── train.py
├── predict.py
├── utils.py
├── main.py
├── requirements.txt
└── runs/                 # создаётся автоматически
```

## Установка
```bash
conda create -n molmamba_radstab python=3.10 -y
conda activate molmamba_radstab
pip install -r requirements.txt
```

## Формат данных

Входной CSV (пример data.csv):
- smiles — SMILES молекулы (строка, обязательно)
- solvent_smiles — SMILES растворителя (строка, опционально, не используется в модели)
- diel_const — диэлектрическая проницаемость растворителя (float)
- concentration — концентрация, моль/л (float)
- dose_constant — целевой dose constant (float), опционален для инференса, при наличии будет рассчитан `abs_error`

Минимальный набор для предсказания: smiles, diel_const, concentration.

## Предобучение (SSL)

Две стадии: SDA (CORAL-выравнивание атом/фрагмент) и Masked Fusion (реконструкция молекулярных дескрипторов). Предобучение проводится только на молекуле, без использования данных растворителя.

```bash
python ssl_pretrain.py --csv data.csv --batch_size 16 \
  --epochs_sda 50 --epochs_mask 80 \
  --ckpt ssl_pretrained.pt
```

## Обучение с scaffold k-fold CV

```bash
python train.py --csv data.csv --ssl_ckpt ssl_pretrained.pt \
  --folds 5 --epochs 300 --out_dir runs
```

Особенности:
- фолды формируются по **Bemis–Murcko scaffold**;
- стандартизация числовых фич выполняется по train фолду;
- применяется ранняя остановка по `val_RMSE`;
- сохраняются модели `runs/fold{k}_best.pt` и отчёты `runs/cv_summary.json`.

## Предсказания

Для одного чекпоинта:
```bash
python predict.py --csv data.csv --ckpt runs/fold1_best.pt --out_csv preds.csv
```

Ансамбль по всем фолдам в директории:
```bash
python predict.py --csv data.csv --ckpt runs/ --out_csv preds_ens.csv
```

Опционально можно сохранить эмбеддинги:
```bash
python predict.py --csv data.csv --ckpt runs/ --out_csv preds.csv \
  --save_embeddings --embeddings_path embeds.npz
```

Выходной CSV содержит предсказания `pred_dc`; при наличии `dose_constant` дополнительно рассчитывается `abs_error`.

## Гиперпараметры

| Параметр          | Значение по умолчанию |
|-------------------|-----------------------|
| `d_model`         | 256                   |
| `gnn_layers`      | 3                     |
| `dropout`         | 0.1                   |
| `lr`              | 2e-4                  |
| `weight_decay`    | 1e-4                  |
| `folds`           | 5                     |

## Метрики

- **RMSE**, **MAE**, **R²** — вычисляются по каждому фолду и по итогам CV.
- Отчёты сохраняются в:
  - `runs/fold{k}_metrics.json`
  - `runs/cv_summary.json`
