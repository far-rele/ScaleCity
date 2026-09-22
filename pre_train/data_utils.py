# data_utils.py
import os
import ast
import pandas as pd
import numpy as np
import h3
import pickle
import torch
from torch.utils.data import Dataset
from PIL import Image
from config import Config


def load_and_clean_data():
    """Load all CSV/PKL inputs, clean them, and return the valid grids and data dicts."""
    print(">>> Loading and cleaning data...")

    def clean_id(x):
        return str(x).strip()

    # 1. H3 master list
    df_h3 = pd.read_csv(Config.H3_ID_FILE, header=None, names=['h3_id'], dtype=str)
    df_h3['h3_id'] = df_h3['h3_id'].apply(clean_id)

    # 2. SVI mapping
    df_svi = pd.read_csv(Config.SVI_MAPPING_FILE, dtype=str)
    svi_id_col = df_svi.columns[0]
    df_svi[svi_id_col] = df_svi[svi_id_col].apply(clean_id)

    # 3. POI mapping
    df_poi_map = pd.read_csv(Config.POI_MAPPING_FILE, dtype=str)
    poi_id_col = df_poi_map.columns[0]
    df_poi_map[poi_id_col] = df_poi_map[poi_id_col].apply(clean_id)

    print("Loading POI text...")
    df_poi_text = pd.read_csv(Config.POI_TEXT_FILE, dtype=str)
    poi_text_dict = pd.Series(
        df_poi_text.iloc[:, 1].values,
        index=df_poi_text.iloc[:, 0].apply(clean_id).values
    ).to_dict()

    # 4. AlphaEarth embeddings (stored as a DataFrame with columns ['h3_id', 'embedding'])
    print("Loading AlphaEarth embeddings...")
    try:
        df_ae = pd.read_pickle(Config.AE_EMBED_FILE)
    except Exception:
        with open(Config.AE_EMBED_FILE, 'rb') as f:
            df_ae = pickle.load(f)

    ae_data = {}
    if 'h3_id' not in df_ae.columns or 'embedding' not in df_ae.columns:
        raise ValueError(f"AE PKL columns mismatch, found: {df_ae.columns}")

    for _, row in df_ae.iterrows():
        hid = clean_id(row['h3_id'])
        emb = np.array(row['embedding'], dtype=np.float32)
        ae_data[hid] = emb

    print(f"Converted AE data, count: {len(ae_data)}")

    # --- Build per-grid dicts ---
    def parse_list(x):
        try:
            if isinstance(x, list):
                return x
            l = ast.literal_eval(x)
            return l if isinstance(l, list) else []
        except Exception:
            return []

    # SVI
    svi_dict = {}
    for _, row in df_svi.iterrows():
        hid = row[svi_id_col]
        imgs = parse_list(row.iloc[1])
        if imgs:
            svi_dict[hid] = imgs

    # POI
    poi_dict = {}
    for _, row in df_poi_map.iterrows():
        hid = row[poi_id_col]
        p_ids = parse_list(row.iloc[1])
        texts = [poi_text_dict.get(str(pid).strip(), "") for pid in p_ids]
        texts = [t for t in texts if t]
        if texts:
            poi_dict[hid] = texts

    print("\n--- Sample keys ---")
    if len(svi_dict) > 0:
        print(f"SVI sample key: '{list(svi_dict.keys())[0]}'")
    if len(poi_dict) > 0:
        print(f"POI sample key: '{list(poi_dict.keys())[0]}'")
    if len(ae_data) > 0:
        print(f"AE  sample key: '{list(ae_data.keys())[0]}'")

    # 5. Intersection of all three modalities
    valid_h3 = set(svi_dict.keys()) & set(poi_dict.keys()) & set(ae_data.keys())
    valid_h3 = list(valid_h3)

    print(f"\nRaw H3 grids: {len(df_h3)}")
    print(f"Grids with SVI: {len(svi_dict)}")
    print(f"Grids with POI: {len(poi_dict)}")
    print(f"Grids with AE:  {len(ae_data)}")
    print(f"=== Final valid grids (intersection): {len(valid_h3)} ===")

    if len(valid_h3) == 0:
        raise ValueError("No valid grids! Check that the sample keys above are consistently formatted.")

    # 6. Build neighbor graph (1-hop edges only)
    print("Building neighbor graph...")
    adj_dict = {}
    valid_set = set(valid_h3)

    for hid in valid_h3:
        try:
            neighbors = h3.k_ring(hid, 1)
            valid_neighbors = [n for n in neighbors if n in valid_set]
            adj_dict[hid] = valid_neighbors
        except Exception:
            continue

    return valid_h3, svi_dict, poi_dict, ae_data, adj_dict


class UrbanDataset(Dataset):
    """Returns H3 IDs only; the Collator loads the actual data to enable dynamic
    subgraph sampling.
    """
    def __init__(self, h3_ids):
        self.h3_ids = h3_ids

    def __len__(self):
        return len(self.h3_ids)

    def __getitem__(self, idx):
        return self.h3_ids[idx]


class UrbanGraphCollator:
    """Expands a batch of center nodes into a subgraph and loads all related data."""
    def __init__(self, svi_dict, poi_dict, ae_data, adj_dict, processor_img, tokenizer_txt):
        self.svi_dict = svi_dict
        self.poi_dict = poi_dict
        self.ae_data = ae_data
        self.adj_dict = adj_dict

        self.processor_img = processor_img
        self.tokenizer_txt = tokenizer_txt

    def __call__(self, batch_h3_ids):
        # 1. Subgraph expansion: BFS over GAT_NEIGHBOR_DEPTH hops so layer-k GAT
        #    can reach k-hop nodes. adj_dict stores only 1-hop edges.
        involved_nodes = set(batch_h3_ids)
        frontier = set(batch_h3_ids)
        for _ in range(Config.GAT_NEIGHBOR_DEPTH):
            next_frontier = set()
            for hid in frontier:
                for nbr in self.adj_dict.get(hid, []):
                    if nbr not in involved_nodes:
                        next_frontier.add(nbr)
            involved_nodes.update(next_frontier)
            frontier = next_frontier

        unique_nodes = list(involved_nodes)
        node_to_idx = {hid: i for i, hid in enumerate(unique_nodes)}
        num_nodes = len(unique_nodes)

        # 2. Build subgraph adjacency matrix [num_nodes, num_nodes] (with self-loops)
        adj_matrix = torch.eye(num_nodes)

        for hid in unique_nodes:
            curr_idx = node_to_idx[hid]
            neighbors = self.adj_dict.get(hid, [])
            for n_hid in neighbors:
                if n_hid in node_to_idx:
                    nb_idx = node_to_idx[n_hid]
                    adj_matrix[curr_idx, nb_idx] = 1.0
                    adj_matrix[nb_idx, curr_idx] = 1.0  # undirected

        # 3. Load data
        batch_images = []
        batch_texts = []
        batch_ae = []
        batch_svi_masks = []
        batch_poi_masks = []

        for hid in unique_nodes:
            # --- AE ---
            batch_ae.append(self.ae_data[hid])

            # --- SVI ---
            img_files = self.svi_dict.get(hid, [])
            if len(img_files) > Config.MAX_IMAGES_PER_GRID:
                img_files = np.random.choice(img_files, Config.MAX_IMAGES_PER_GRID, replace=False)

            pixel_values_list = []
            for img_name in img_files:
                img_path = os.path.join(Config.SVI_IMAGE_DIR, img_name)
                try:
                    image = Image.open(img_path).convert("RGB")
                    inputs = self.processor_img(images=image, return_tensors="pt")
                    pixel_values_list.append(inputs['pixel_values'].squeeze(0))  # [3, 224, 224]
                except Exception as e:
                    print(f"Failed to read SVI image {img_name}, skipped:", e)
                    continue

            # Generate mask and zero-pad to MAX_IMAGES_PER_GRID
            current_count = len(pixel_values_list)
            mask = torch.zeros(Config.MAX_IMAGES_PER_GRID, dtype=torch.float32)
            mask[:current_count] = 1.0
            batch_svi_masks.append(mask)
            if current_count == 0:
                pixel_values_list.append(torch.zeros(3, 224, 224))
            while len(pixel_values_list) < Config.MAX_IMAGES_PER_GRID:
                pixel_values_list.append(torch.zeros(3, 224, 224))
            grid_imgs = torch.stack(pixel_values_list[:Config.MAX_IMAGES_PER_GRID])
            batch_images.append(grid_imgs)

            # --- POI ---
            texts = self.poi_dict.get(hid, [])
            if len(texts) > Config.MAX_POIS_PER_GRID:
                texts = np.random.choice(texts, Config.MAX_POIS_PER_GRID, replace=False).tolist()
            # Generate POI mask and pad with empty strings (masked out during aggregation)
            current_poi_count = len(texts)
            p_mask = torch.zeros(Config.MAX_POIS_PER_GRID, dtype=torch.float32)
            p_mask[:current_poi_count] = 1.0
            batch_poi_masks.append(p_mask)
            if len(texts) == 0:
                texts = [""]
            while len(texts) < Config.MAX_POIS_PER_GRID:
                texts.append("")
            batch_texts.extend(texts[:Config.MAX_POIS_PER_GRID])  # flatten for the tokenizer

        # 4. Tensorize
        # SVI: [Total_Nodes, MAX_IMG, 3, 224, 224]
        tensor_svi = torch.stack(batch_images)
        tensor_svi_mask = torch.stack(batch_svi_masks)

        # POI: tokenize all texts at once, then reshape
        tokenized_poi = self.tokenizer_txt(
            batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
        )
        poi_shape = (num_nodes, Config.MAX_POIS_PER_GRID, -1)
        tensor_poi_ids = tokenized_poi['input_ids'].view(*poi_shape)
        tensor_poi_mask = tokenized_poi['attention_mask'].view(*poi_shape)
        tensor_poi_grid_mask = torch.stack(batch_poi_masks)

        # AE: [Total_Nodes, AE_Dim]
        tensor_ae = torch.tensor(np.array(batch_ae), dtype=torch.float32)

        # Indices of the original center nodes within the subgraph
        target_indices = [node_to_idx[hid] for hid in batch_h3_ids]
        target_indices = torch.tensor(target_indices, dtype=torch.long)

        return {
            'svi': tensor_svi,                      # [N, M_img, 3, H, W]
            'svi_mask': tensor_svi_mask,            # [N, M_img]
            'poi_ids': tensor_poi_ids,              # [N, M_poi, L]
            'poi_mask': tensor_poi_mask,            # [N, M_poi, L] (BERT mask)
            'poi_grid_mask': tensor_poi_grid_mask,  # [N, M_poi] (aggregation mask)
            'ae': tensor_ae,                        # [N, D_ae]
            'adj': adj_matrix,                      # [N, N]
            'target_idx': target_indices            # [B]
        }
