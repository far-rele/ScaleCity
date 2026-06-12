import torch
from torch.utils.data import Dataset
import pandas as pd
import geopandas as gpd
import numpy as np
import random
import pickle
from config import Config


class UrbanDataset(Dataset):
    def __init__(self, mode='train', spatial_mappings=None, specific_base_ids=None, tasks_config=None, weights_config=None):
        """
        Args:
            mode: 'train' or 'test'
            spatial_mappings: spatial mapping dict (from utils.build_spatial_mappings)
            specific_base_ids: (list, optional) base-scale IDs from the K-Fold split
            tasks_config: (dict, optional) task config. Defaults to Config.TASKS.
            weights_config: (dict, optional) sample weights. Defaults to Config.SAMPLE_WEIGHTS.
        """
        self.mode = mode
        if spatial_mappings is None:
            raise ValueError("spatial_mappings is required.")

        self.spatial_mappings = spatial_mappings
        self.base_scale = spatial_mappings.get('base_scale', Config.BASE_SCALE)

        self.tasks_config = tasks_config if tasks_config is not None else Config.TASKS
        self.weights_config = weights_config if weights_config is not None else Config.SAMPLE_WEIGHTS

        # 1. Load embeddings
        with open(Config.EMBEDDING_PATH, 'rb') as f:
            self.emb_dict = pickle.load(f)

        self.valid_h3_list = list(self.emb_dict.keys())
        self.h3_to_idx = {h3_id: i for i, h3_id in enumerate(self.valid_h3_list)}

        # Build embedding matrix [N, 3, 512]
        emb_matrix = []
        for h3id in self.valid_h3_list:
            feats = self.emb_dict[h3id]
            emb_matrix.append(np.stack([feats['svi_emb'], feats['poi_emb'], feats['ae_emb']]))
        self.embedding_matrix = torch.tensor(np.array(emb_matrix), dtype=torch.float32)

        # 2. Determine target base IDs
        if specific_base_ids is not None:
            self.target_base_ids = set(specific_base_ids)
        else:
            # Fallback random split (mainly for debugging)
            if self.base_scale == 'h3':
                all_ids = self.valid_h3_list
            else:
                base_overlap = self.spatial_mappings['overlap'].get(self.base_scale, {})
                all_ids = list(base_overlap.keys())

            all_ids.sort()
            random.seed(Config.SPLIT_SEED)
            random.shuffle(all_ids)

            split_idx = int(len(all_ids) * Config.TRAIN_RATIO)
            if mode == 'train':
                self.target_base_ids = set(all_ids[:split_idx])
            else:
                self.target_base_ids = set(all_ids[split_idx:])

        # 3. Load and filter task data
        self.samples = []
        self.sample_weights = []

        self._load_tasks()

    def _load_tasks(self):
        """Load samples per task. For each sample, find its base-scale parent and
        keep it only if that parent is in self.target_base_ids. IDs are coerced to str.
        """
        for task_name, task_cfg in self.tasks_config.items():
            sample_w = self.weights_config.get(task_name, 1.0)
            t_type = task_cfg['type']
            id_col = task_cfg['id_col']
            target_col = task_cfg['target_col']

            # Case A: H3 task
            if t_type == 'h3_csv':
                df = pd.read_csv(task_cfg['path'])
                df[id_col] = df[id_col].astype(str)

                parent_mapping = self.spatial_mappings['h3_parent']

                for _, row in df.iterrows():
                    h3_id = row[id_col]
                    label = row[target_col]

                    if pd.isna(label):
                        continue

                    if h3_id not in self.h3_to_idx:
                        continue

                    base_parent_id = parent_mapping.get(h3_id)

                    # Must have a parent in the current train/test split
                    if base_parent_id is None or base_parent_id not in self.target_base_ids:
                        continue

                    self.samples.append({
                        'task_name': task_name,
                        'indices': [self.h3_to_idx[h3_id]],
                        'weights': [1.0],
                        'label': float(label)
                    })
                    self.sample_weights.append(sample_w)

            # Case B: polygon task (street / grid)
            elif t_type in ['street_shp', 'grid_shp']:

                if t_type == 'street_shp':
                    scale_key = task_cfg.get('scale_key', 'street_normal')
                else:
                    scale_key = task_cfg['scale_key']

                shp_path = task_cfg['path']

                overlap_map = self.spatial_mappings['overlap'].get(scale_key, {})

                is_base_scale = (scale_key == self.base_scale)
                parent_map = {}
                if not is_base_scale:
                    parent_map = self.spatial_mappings['parent'].get(scale_key, {})

                gdf = gpd.read_file(shp_path)
                gdf[id_col] = gdf[id_col].astype(str)

                for _, row in gdf.iterrows():
                    poly_id = row[id_col]
                    label = row[target_col]

                    if pd.isna(label):
                        continue

                    if is_base_scale:
                        if poly_id not in self.target_base_ids:
                            continue
                    else:
                        base_parent_id = parent_map.get(poly_id)
                        if base_parent_id is None or base_parent_id not in self.target_base_ids:
                            continue

                    # Build sample from H3 overlap
                    if poly_id in overlap_map:
                        grid_info = overlap_map[poly_id]
                        indices = []
                        weights = []

                        for h3_id, w in grid_info:
                            if h3_id in self.h3_to_idx:
                                indices.append(self.h3_to_idx[h3_id])
                                weights.append(w)

                        if len(indices) > 0:
                            self.samples.append({
                                'task_name': task_name,
                                'indices': indices,
                                'weights': weights,
                                'label': float(label)
                            })
                            self.sample_weights.append(sample_w)

    def get_weights(self):
        return self.sample_weights

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        indices = torch.tensor(sample['indices'], dtype=torch.long)
        feats = self.embedding_matrix[indices]
        weights = torch.tensor(sample['weights'], dtype=torch.float32)
        label = torch.tensor(sample['label'], dtype=torch.float32)

        return {
            'task_name': sample['task_name'],
            'feats': feats,
            'weights': weights,
            'label': label
        }


def urban_collate_fn(batch):
    batch_task_names = [item['task_name'] for item in batch]
    batch_labels = torch.stack([item['label'] for item in batch])

    max_len = max([item['feats'].shape[0] for item in batch])
    feat_dim = batch[0]['feats'].shape[2]

    batch_feats = torch.zeros((len(batch), max_len, 3, feat_dim), dtype=torch.float32)
    batch_weights = torch.zeros((len(batch), max_len), dtype=torch.float32)
    batch_masks = torch.zeros((len(batch), max_len), dtype=torch.float32)

    for i, item in enumerate(batch):
        curr_len = item['feats'].shape[0]
        batch_feats[i, :curr_len, :, :] = item['feats']
        batch_weights[i, :curr_len] = item['weights']
        batch_masks[i, :curr_len] = 1.0

    return {
        'task_names': batch_task_names,
        'feats': batch_feats,
        'weights': batch_weights,
        'masks': batch_masks,
        'labels': batch_labels
    }
