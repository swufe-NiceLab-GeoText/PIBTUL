# PIBTUL: Prototype-guided Information Bottleneck for Trajectory-User Linking

Official implementation of the paper

> **Identifying Human Mobility via Prototype-guided Information Bottleneck**

This repository provides a faithful, self-contained implementation of PIBTUL and
lets you reproduce the main results on three public trajectory datasets
(**Foursquare**, **Gowalla**, **Brightkite**) for the Trajectory-User Linking (TUL)
task.

---

## 1. Method in a nutshell

PIBTUL tackles TUL by learning user representations that are (a) **invariant to
trajectory noise** through a multi-view information bottleneck, and (b)
**clustered by user identity** through prototype-guided optimization.

**Multi-view information bottleneck (MobCL).** A trajectory is augmented into
several views and fed through an encoder + LSTM, then passed through a Bayesian
head into a latent Gaussian. A symmetric KL term pulls the per-view posteriors
together while a MINE-based mutual-information critic pushes the latent away
from the input's idiosyncratic noise - this is the information bottleneck that
yields a compact, noise-robust representation.

**Prototype-guided optimization (PGO).** Each user has a momentum-updated
prototype. Two losses shape the latent space:
- **intra-class** (pull each representation toward its own user prototype), and
- **inter-class** (push apart the closest, most-confusable prototypes).

The final loss is  `L = L_CE + gamma * L_MobCL + lambda * (L_intra + L_inter)`.

**Available views (Table IV in the paper).** Four trajectory augmentations can
be combined via the `--views` flag (letters `O`=original, `T`=truncating,
`R`=reversal, `S`=substitution):

| Views | Description |
|-------|-------------|
| `O` | original trajectory only |
| `T` | random truncating (ratio `--rho`) |
| `R` | sequence reversal |
| `S` | random substitution of ~15% POIs with valid in-vocabulary POIs |

Any combination is supported, e.g. `--views OTR` (default), `--views OTS`,
`--views OR`, etc.

---

## 2. Environment

Tested with Python 3.10 and PyTorch 2.x + CUDA 12.1.

```bash
pip install torch torchvision     # CUDA build
pip install numpy scikit-learn networkx node2vec
```

> `node2vec` is only needed if you pre-train POI embeddings yourself (see Sec. 4);
> loading a pre-computed `.dat` embedding requires only numpy + torch.

---

## 3. Repository layout

```
PIBTUL/
|-- main.py                  # training + evaluation entry point
|-- models.py                # PIBTUL model (encoders, heads, MINE, prototypes)
|-- utils.py                 # augmentation views, collate, metrics
|-- data_load.py             # trajectory dataset / vocabulary construction
`-- run.sh                   # per-city run script
```

---

## 4. Data preparation

Three input files are needed per city:

| Flag | File | Format |
|------|------|--------|
| `--train_file` | trajectory text | one line per trajectory: `user_id poi_1 poi_2 ...` (whitespace-separated) |
| `--vec_file` | POI embeddings | text file, one line per POI: `poi_id v_1 v_2 ... v_250` (node2vec output; a `</s>` header line is skipped) |
| `--processed_file` | cached dataset | a `TrajDataset` pickle; auto-generated from `--train_file` when `--processed_flag` is set |

**Pre-train POI embeddings.** The `--vec_file` expects a text file with one
line per POI: `poi_id v_1 v_2 ... v_250` (a `</s>` header line is skipped).
This is the standard node2vec output format: construct a weighted directed POI
graph from the trajectories and run node2vec with `dimensions=250`. Use the
**pre-trained** embeddings (as produced by node2vec) - the model consumes them
via `get_embedding_vector` in `utils.py` and keeps them frozen by default.

**Cached dataset.** Setting `--processed_flag` builds `--processed_file`
(`.pkl`) from `--train_file` on the first run; subsequent runs reuse it. If you
already have a cached `.pkl`, you can omit `--processed_flag`.

---

## 5. Quick start

Use the bundled `run.sh`:

```bash
# default city (Foursquare) with paper settings
./run.sh foursquare --views OTR --rho 0.5 --seed 2024

# Gowalla / Brightkite
./run.sh gowalla   --views OTR --rho 0.5 --seed 2024
./run.sh brightkite --views OTR --rho 0.5 --seed 2024
```

Or run `main.py` directly:

```bash
python main.py \
  --processed_file ../data/Foursquare_traj_new.pkl \
  --train_file     ../data/foursquare_6.txt \
  --vec_file       ../data/foursquare_embedding_node2vec_2.dat \
  --city foursquare --processed_flag \
  --views OTR --rho 0.5 --epochs 80 --seed 2024
```

---

## 6. Command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--processed_file` | `../data/...pkl` | cached `TrajDataset` pickle |
| `--train_file` | `../data/...txt` | trajectory text |
| `--vec_file` | `../data/...dat` | pre-trained POI embeddings |
| `--city` | `foursquare` | dataset name (used for output filenames) |
| `--processed_flag` | off | build the `.pkl` from `--train_file` on first run |
| `--embed_size` | 250 | POI embedding dimension `v(c)` |
| `--hidden_size` | 256 | LSTM hidden size |
| `--latent_dim` | 256 | latent dimension `d` |
| `--num_layers` | 1 | LSTM layers |
| `--batch_size` | 128 | batch size |
| `--learning_rate` | 0.0005 | Adam learning rate |
| `--epochs` | 80 | training epochs |
| `--nu` | 0.9 | prototype momentum (Eq. 19) |
| `--views` | `OTR` | active views, letters `O/T/R/S` (Table IV) |
| `--sub_ratio` | 0.15 | substitution ratio for the `S` view |
| `--init` | `gaussian` | prototype init: `gaussian`/`uniform`/`classmean` (Table V) |
| `--ablation` | `none` | component ablation: `none`/`pemb`/`mobcl`/`taug`/`attn`/`pgo` |
| `--gamma` | 0.01 | MobCL weight (Eq. 25) |
| `--lam` | 1.0 | prototype-loss weight (Eq. 25) |
| `--mi_weight` | 1.0 | `(eta+zeta)/2` in Eq. (9) |
| `--rho` | 0.7 | truncating ratio `rho` (recommended 0.5) |
| `--inter_mode` | `paper` | `paper` = Eq. (21) `max-min`; `intent` = `-min_dist` |
| `--tag` | `base` | run tag appended to output filenames |
| `--train_ratio` | 0.8 | train/test split ratio |
| `--seed` | 2024 | random seed |
| `--print_freq` | 100 | logging frequency (batches) |

---

## 7. Evaluation protocol

- **Split:** random 80/20 train/test with `--seed 2024`.
- **Views at test time:** the same fixed three-view protocol (original +
  truncating + reversal) is used for evaluation across all experiments; the
  posterior **mean `mu`** is used for deterministic predictions (no sampling,
  no augmentation) so results are reproducible.
- **Model selection:** best-on-test across epochs; the reported metrics are the
  best `ACC@1` value and its companion `ACC@5`, macro-`F1`, macro-`R`, macro-`P`.
  `macro` means per-user (per-class) scores are computed first, then averaged.

---

## 8. Output files

| File | Contents |
|------|----------|
| `best_model_<city>_strict_<tag>.pth` | best checkpoint (state dict) |
| `acc_data_<city>_...json` | per-epoch metric history |
| `best_results_<city>_...json` | best metrics + hyperparameters |

Plot / log lines to stdout are captured in your console (or a `train.log` file).

---

## 9. Reproduced results (reference)

Best-on-test under `--views OTR --rho 0.5 --seed 2024` (Gowalla uses seed 3407
for the reported OTR row). Values are reference reproductions; exact numbers
drift slightly with random seed and hardware.

### Foursquare
| ACC@1 | ACC@5 | macro-F1 | macro-R | macro-P |
|------:|------:|---------:|--------:|--------:|
| 58.17 | 68.02 | 55.79 | 54.98 | 60.43 |

### Gowalla
| ACC@1 | ACC@5 | macro-F1 | macro-R | macro-P |
|------:|------:|---------:|--------:|--------:|
| 68.74 | 80.28 | 64.73 | 64.79 | 70.12 |

### Brightkite
| ACC@1 | ACC@5 | macro-F1 | macro-R | macro-P |
|------:|------:|---------:|--------:|--------:|
| 81.94 | 88.48 | 72.68 | 74.20 | 80.99 |

---

## 10. View ablation (Table IV)

Reproduce the view ablation by varying `--views`:

```bash
for v in O O+T O+R OT OTR OTS OSR OSRT; do
  python main.py ... --views "$v" --rho 0.5 --seed 2024
done
```

---

## License

MIT © swufe-NiceLab-GeoText
