# ScaleCity: Multi-Scale Urban Region Representation Learning via a Prompt-Conditioned Hypernetwork

This is the official implementation of **ScaleCity**.

ScaleCity is a framework for urban multi-scale representation learning and downstream
indicator prediction. It learns city H3-cell embeddings via multimodal contrastive
pretraining, then predicts a range of urban indicators (population, air quality, GDP,
housing price, shared-bike ridership, etc.) across multiple spatial scales (road-network
blocks L1–L5 / L11, and regular grids 1/3/5 km) using prompt-conditioned,
multi-task learning.

The framework consists of two components:

1. **Pretraining** (`pre_train/`): three-modality contrastive learning that produces
   city H3-cell embeddings.
2. **Downstream learning** (`prompt_learning/`): prompt-conditioned multi-task prediction
   built on top of the pretrained embeddings.

![Framework](figure/framework.png)

## Project structure

```
ScaleCity/
├── pre_train/                  # Multimodal contrastive pretraining
│   ├── config.py               # paths, model names, GPUs, hyperparameters
│   ├── models.py               # UrbanGraphModel: 3-modality encoders + GAT + projection
│   ├── data_utils.py           # data loading, UrbanDataset, UrbanGraphCollator
│   ├── train_torchrun.py       # DDP training entry point (torchrun)
│   └── output/                 # pretrained H3-cell embeddings
│
└── prompt_learning/            # Downstream prediction
    ├── config.py               # main config: city, embedding path, tasks
    ├── model.py                # UrbanPromptModel (dual-stream + HyperNetwork)
    ├── dataset.py              # UrbanDataset, urban_collate_fn
    ├── utils.py                # spatial mapping, scaler, prompt encoding, metrics
    ├── train.py                # K-fold training
    └── config/
        ├── TASKS.json          # task registry (task name -> data file)
        └── SCALE_DEFINITIONS.json  # scale definitions
```

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.9+ and PyTorch 2.x. A CUDA-capable GPU is recommended.

Encoder weights (CLIP, text encoders) are public HuggingFace models and are downloaded
automatically on first run: `openai/clip-vit-base-patch32`, `hfl/chinese-macbert-base`
(pretraining) and `BAAI/bge-base-zh-v1.5` (prompt learning). If automatic downloading
fails, download the model folder locally and point the corresponding environment
variable to it before running — `SCALECITY_BERT_PATH` for prompt learning,
`SCALECITY_CLIP_PATH` / `SCALECITY_TEXT_ENCODER` for pretraining, e.g.:

```bash
SCALECITY_BERT_PATH=/path/to/bge-base-zh-v1.5 python train.py
```

## Usage

### Pretraining

> **Data required (not included).** Due to size and licensing constraints, this
> repository does **not** contain the pretraining inputs. To run pretraining on your
> own city, prepare the following data and set the corresponding paths in
> `pre_train/config.py`:
>
> | Input | Config entry | Description |
> |---|---|---|
> | H3 cell ids | `H3_ID_FILE` | CSV of H3 cell ids covering the city (resolution 8) |
> | SVI images + mapping | `SVI_IMAGE_DIR`, `SVI_MAPPING_FILE` | street-view images and the H3-to-image mapping CSV |
> | POI text + mapping | `POI_TEXT_FILE`, `POI_MAPPING_FILE` | POI textual descriptions and the H3-to-POI mapping CSV |
> | AE embeddings | `AE_EMBED_FILE` | AlphaEarth 64-d embeddings per H3 cell (pkl) |

```bash
cd pre_train

# Multi-GPU training (set the GPU count with --nproc_per_node)
torchrun --nproc_per_node=4 train_torchrun.py

# Single-GPU debugging
python train_torchrun.py
```

Output: `pre_train/output/{CITYNAME}/{timestamp}/{timestamp}.pkl` (H3-cell embeddings).

### Downstream training

> **Ready to run out of the box.** Everything needed for downstream training on
> Shenzhen is included in this repository:
>
> - the pretrained H3-cell embeddings (`pre_train/output/shenzhen/shenzhen_embedding.pkl`);
> - the task shapefiles for all 72 active tasks (8 indicator families x 9 spatial
>   scales) under `data/task/`.
>
> The prompt encoder (`BAAI/bge-base-zh-v1.5`) is downloaded automatically from the
> HuggingFace Hub on first run, so with an internet connection and a CUDA GPU you can
> start training directly — no extra data preparation is needed:

```bash
cd prompt_learning

# K-fold cross-validation training
python train.py
```

## Configuration

Main settings live in `prompt_learning/config.py` (downstream) and
`pre_train/config.py` (pretraining).

