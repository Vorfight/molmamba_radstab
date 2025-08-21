# Mol-Mamba для предсказания радиационной стабильности (dose constant)

Полный пайплайн для регрессии **dose constant** из:
- атомного графа молекулы (узлы=атомы, рёбра=связи) → **Mol-Mamba encoder** (GNN → упорядочивание → Mamba/GRU);
- фрагментного графа (кольцевые системы) → **Frag GNN encoder**;
- числовых фич (среда: `diel_const`, `conc_stand` + RDKit-дескрипторы молекулы);
- **Mamba-Transformer fuser** для слияния `[ATOM | FRAG | NUM]`;
- двухступенчатого SSL-предобучения (**SDA + e-semantic masked fusion**) и супервизед-обучения с **scaffold k-fold CV**.

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
- smiles — SMILES молекулы (строка, обязателен)
- solv_smiles — SMILES растворителя (строка, опционально — пока не используется)
- diel_const — диэлектрическая проницаемость растворителя (float)
- conc_stand — концентрация, моль/л (float)
- dc_stand — целевой dose constant (float), опционален для инференса

Минимум для предсказания: smiles, diel_const, conc_stand.

## Предобучение (SSL)

Две стадии: SDA (CORAL-выравнивание атом/фрагмент) и Masked Fusion (реконструкция NUM-фич).
```bash
python ssl_pretrain.py --csv data.csv --batch_size 16 \
  --epochs_sda 50 --epochs_mask 80 \
  --ckpt ssl_pretrained.pt
```
Скрипт сохранит веса модели и параметры стандартизации числовых фич.

## Обучение с scaffold k-fold CV
```bash
python train.py --csv data.csv --ssl_ckpt ssl_pretrained.pt \
  --folds 5 --epochs 300 --out_dir runs
```
Что делает:
- фолды по **Bemis–Murcko scaffold** (честная валидация по каркасам);
- стандартизация числовых фич **по train фолда**;
- early stopping по `val_RMSE`;
- сохраняет `runs/fold{k}_best.pt` и `runs/cv_summary.json`.

Полезные флаги:
- `--d_model 128` и/или `--dropout 0.2-0.4` для маленьких датасетов (у тебя 44 образца);
- `--folds 11` (мелкие scaffold-группы) — как LOO по каркасам;
- `--weight_decay 1e-3…1e-2`, `--patience 15-25` для жёстче ранней остановки.

## Предсказания

Один чекпоинт:
```bash
python predict.py --csv data.csv --ckpt runs/fold1_best.pt --out_csv preds.csv
```
Ансамбль по всем фолдам в директории:
```bash
python predict.py --csv data.csv --ckpt runs/ --out_csv preds_ens.csv
```
Опционально сохранить эмбеддинги:
```bash
python predict.py --csv data.csv --ckpt runs/ --out_csv preds.csv \
  --save_embeddings --embeddings_path embeds.npz
```
Выходной CSV добавит pred_dc; если есть dc_stand, также abs_error.


## Гиперпараметры (рекомендации)

| Параметр          | Значение по умолчанию | Заметки |
|-------------------|-----------------------|---------|
| `d_model`         | 256 (или 128)         | 128 на маленьком датасете — стабильнее |
| `gnn_layers`      | 3                     | 2–4 обычно достаточно |
| `dropout`         | 0.1 (0.2–0.4)         | Увеличить для 44 образцов |
| `lr`              | 2e-4                  | AdamW, cosine scheduler |
| `weight_decay`    | 1e-4 (1e-3…1e-2)      | Сильнее L2 = меньше переобучения |
| `folds`           | 5 (или 11)            | Scaffold-CV |


## Метрики

- **RMSE**, **MAE**, **R²** — печатаются по каждому фолду и итогу CV.  
- Отчёты сохраняются в:
  - `runs/fold{k}_metrics.json`
  - `runs/cv_summary.json`


