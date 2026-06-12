import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import h3
import pickle
import json
import math
from tqdm import tqdm
from shapely.geometry import Polygon
from transformers import AutoTokenizer, AutoModel
from config import Config
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.style.use('seaborn-v0_8-whitegrid')


# =========================================================
#  0. Config loading from JSON
# =========================================================

def load_config_from_json(config_class):
    """Load JSON config files into the Config class."""
    config_dir = os.path.join(config_class.ROOT_DIR, "prompt_learning/config")

    json_files = {
        'SCALE_DEFINITIONS': 'SCALE_DEFINITIONS.json',
        'TASKS': 'TASKS.json'
    }

    for attr_name, filename in json_files.items():
        filepath = os.path.join(config_dir, filename)
        if not os.path.exists(filepath):
            print(f"[Warning] Config file not found: {filepath}")
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        data = _replace_placeholders(data, config_class.ROOT_DIR, config_class.CITYNAME)

        setattr(config_class, attr_name, data)
        print(f"[Config] Loaded {attr_name} from {filename}")


def load_script_config(filename, root_dir, cityname):
    """Load a script-specific config file (.py or .json) and resolve placeholders.

    Args:
        filename: e.g. config_finetune.py, config_inference.py
        root_dir: root directory for placeholder resolution
        cityname: city name for placeholder resolution
    Returns:
        Config dict with placeholders resolved.
    """
    # .py files live directly under prompt_learning/, .json files under config/
    if filename.endswith('.py'):
        config_dir = os.path.join(root_dir, "prompt_learning")
    else:
        config_dir = os.path.join(root_dir, "prompt_learning/config")
    filepath = os.path.join(config_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[Error] Config file not found: {filepath}")

    if filename.endswith('.py'):
        import importlib.util
        module_name = os.path.splitext(filename)[0]
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        data = module.CONFIG
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

    data = _replace_placeholders(data, root_dir, cityname)
    print(f"[Config] Loaded script config from {filename}")
    return data


def _replace_placeholders(data, root_dir, cityname):
    """Recursively replace {CITYNAME} / {ROOT_DIR} placeholders in a config object."""
    if isinstance(data, dict):
        return {k: _replace_placeholders(v, root_dir, cityname) for k, v in data.items()}
    elif isinstance(data, list):
        return [_replace_placeholders(item, root_dir, cityname) for item in data]
    elif isinstance(data, str):
        result = data.replace('{CITYNAME}', cityname)
        result = result.replace('{ROOT_DIR}', root_dir)
        # Prepend ROOT_DIR for relative paths
        if not os.path.isabs(result) and ('/' in result or '\\' in result):
            result = os.path.join(root_dir, result)
        return result
    else:
        return data


# =========================================================
#  1. Spatial mapping construction
# =========================================================

def _compute_overlap_mapping(poly_gdf, poly_id_col, h3_polys, h3_areas, threshold=0.0):
    """Compute overlap weights between polygons (any scale) and H3 cells."""
    if poly_gdf.crs != "EPSG:3857":
        poly_gdf = poly_gdf.to_crs("EPSG:3857")

    intersection = gpd.overlay(h3_polys, poly_gdf, how='intersection')
    intersection['overlap_area'] = intersection.geometry.area

    mapping = {}
    grouped = intersection.groupby(poly_id_col)

    for pid, group in tqdm(grouped, desc="Computing H3 Overlap", leave=False):
        grids_info = []
        total_overlap_ratio = 0.0
        for _, row in group.iterrows():
            h3_id = row['h3_id']
            overlap = row['overlap_area']
            total = h3_areas.get(h3_id, 1.0)
            ratio = min(overlap / total, 1.0)
            if ratio > 0.001:
                grids_info.append((h3_id, ratio))
                total_overlap_ratio += ratio

        if total_overlap_ratio > threshold:
            mapping[pid] = grids_info

    return mapping


def _get_base_scale_geometry(base_scale):
    """Resolve the base scale geometry.

    Strategy:
    1. Look for a task in TASKS with a matching scale_key (reuse its SHP).
    2. Otherwise fall back to SCALE_DEFINITIONS.
    """
    if base_scale == 'h3':
        return {'type': 'h3'}

    # 1. Look in TASKS
    for t_cfg in Config.TASKS.values():
        if t_cfg.get('type') in ['street_shp', 'grid_shp']:
            curr_key = t_cfg.get('scale_key', 'street_normal') if t_cfg['type'] == 'street_shp' else t_cfg['scale_key']

            if curr_key == base_scale:
                print(f"[Base Scale] Matched in TASKS: {curr_key}")

                path = t_cfg.get('path')

                id_col = t_cfg.get('id_col')
                if not id_col:
                    id_col = 'grid_id' if t_cfg.get('type') == 'grid_shp' else 'street_id'

                return {'type': 'shp', 'path': path, 'id_col': id_col}

    # 2. Look in SCALE_DEFINITIONS
    if base_scale in Config.SCALE_DEFINITIONS:
        print(f"[Base Scale] Matched in SCALE_DEFINITIONS: {base_scale}")
        def_cfg = Config.SCALE_DEFINITIONS[base_scale]
        return {'type': 'shp', 'path': def_cfg['path'], 'id_col': def_cfg['id_col']}

    raise ValueError(f"Definition for base scale [{base_scale}] not found. Check TASKS or SCALE_DEFINITIONS.")


def _resolve_parent_mapping(gdf_child, child_id_col, gdf_base, base_id_col, label="Mapping"):
    """Resolve which base unit each child polygon belongs to.

    Uses a unique temporary child id column to avoid column-name collisions during
    overlay/sjoin. Applies three strategies in order: centroid-within, largest
    overlap, then nearest distance.
    """
    parent_map = {}

    SAFE_CHILD_ID = "__safe_child_id_unique__"

    gdf_child_work = gdf_child.copy()
    gdf_child_work[SAFE_CHILD_ID] = gdf_child_work[child_id_col]

    gdf_child_proj = gdf_child_work.to_crs("EPSG:3857")
    gdf_base_proj = gdf_base.to_crs("EPSG:3857")

    # Strategy 1: centroid within
    gdf_child_centroid = gdf_child_proj.copy()
    gdf_child_centroid['geometry'] = gdf_child_centroid.geometry.centroid

    joined_c = gpd.sjoin(gdf_child_centroid, gdf_base_proj, how='left', predicate='within')

    for _, row in joined_c.iterrows():
        if pd.notnull(row[base_id_col]):
            parent_map[row[SAFE_CHILD_ID]] = row[base_id_col]

    all_child_ids = set(gdf_child_work[SAFE_CHILD_ID])
    matched_ids = set(parent_map.keys())
    orphan_ids = list(all_child_ids - matched_ids)

    if not orphan_ids:
        return parent_map

    print(f"  [{label}] {len(orphan_ids)} unmatched units after first pass (out of {len(all_child_ids)}), starting recall...")

    # Strategy 2: largest overlap
    gdf_orphans = gdf_child_proj[gdf_child_proj[SAFE_CHILD_ID].isin(orphan_ids)]

    try:
        intersection = gpd.overlay(gdf_orphans, gdf_base_proj, how='intersection')

        if not intersection.empty:
            intersection['area'] = intersection.geometry.area
            intersection = intersection.sort_values('area', ascending=False)
            intersection = intersection.drop_duplicates(subset=[SAFE_CHILD_ID])

            for _, row in intersection.iterrows():
                parent_map[row[SAFE_CHILD_ID]] = row[base_id_col]
    except Exception as e:
        print(f"  [Warning] Overlap calculation error: {e}")

    # Strategy 3: nearest distance
    current_matched = set(parent_map.keys())
    still_orphan_ids = list(all_child_ids - current_matched)

    if still_orphan_ids:
        gdf_still_orphans = gdf_child_proj[gdf_child_proj[SAFE_CHILD_ID].isin(still_orphan_ids)].copy()
        gdf_still_orphans['geometry'] = gdf_still_orphans.geometry.centroid

        max_dist = getattr(Config, 'ORPHAN_MAX_DISTANCE', 500)

        try:
            joined_n = gpd.sjoin_nearest(
                gdf_still_orphans, gdf_base_proj, how='left',
                max_distance=max_dist, distance_col='dist'
            )
            joined_n = joined_n.drop_duplicates(subset=[SAFE_CHILD_ID])

            for _, row in joined_n.iterrows():
                if pd.notnull(row[base_id_col]):
                    parent_map[row[SAFE_CHILD_ID]] = row[base_id_col]
        except Exception as e:
            print(f"  [Warning] Nearest calculation error: {e}")

    final_orphans = len(all_child_ids) - len(parent_map)
    if final_orphans < len(orphan_ids):
        print(f"  [{label}] Recovered {len(orphan_ids) - final_orphans} units, remaining unmatched: {final_orphans}")

    return parent_map


def build_spatial_mappings():
    """Build all spatial mappings required by the tasks.

    Supports H3 orphan recovery and conversion between arbitrary scales.
    """
    # 1. Load H3 base data
    print("[Preprocess] Loading H3 embedding data to build the spatial base...")
    with open(Config.EMBEDDING_PATH, 'rb') as f:
        emb_data = pickle.load(f)
        print("Using H3 embedding:", Config.EMBEDDING_PATH)
    valid_h3_ids = list(emb_data.keys())

    # Build H3 polygon GDF (H3 cells are treated as areas, not points)
    h3_polys = []
    valid_h3_ids_str = []
    for hid in valid_h3_ids:
        try:
            poly = Polygon(h3.h3_to_geo_boundary(str(hid), geo_json=True))
            h3_polys.append(poly)
            valid_h3_ids_str.append(str(hid))
        except Exception:
            continue

    gdf_h3_poly = gpd.GeoDataFrame({'h3_id': valid_h3_ids_str}, geometry=h3_polys, crs="EPSG:4326")
    h3_areas = gdf_h3_poly.to_crs("EPSG:3857").set_index('h3_id').geometry.area.to_dict()

    # 2. Process the base scale
    base_scale = Config.BASE_SCALE
    print(f"[Preprocess] Base scale: {base_scale}")

    mappings = {
        'base_scale': base_scale,
        'valid_h3_ids': valid_h3_ids_str,
        'h3_parent': {},
        'overlap': {},
        'parent': {},
        'geometric_counts': {}
    }

    base_info = _get_base_scale_geometry(base_scale)
    gdf_base = None
    SAFE_BASE_ID = "__safe_base_id__"

    if base_info['type'] == 'h3':
        gdf_base = gdf_h3_poly.copy()
        gdf_base[SAFE_BASE_ID] = gdf_base['h3_id']
        mappings['geometric_counts'][base_scale] = len(gdf_base)

        # H3 -> H3 belongs to itself
        for hid in valid_h3_ids_str:
            mappings['h3_parent'][hid] = hid

    else:
        path = base_info['path']
        id_col = base_info['id_col']
        if not os.path.exists(path):
            raise FileNotFoundError(f"SHP not found: {path}")

        gdf_base = gpd.read_file(path).to_crs("EPSG:4326")
        gdf_base[SAFE_BASE_ID] = gdf_base[id_col].astype(str)

        mappings['geometric_counts'][base_scale] = len(gdf_base)

        print(f"[Preprocess] Computing base scale ({base_scale}) H3 overlap...")
        mappings['overlap'][base_scale] = _compute_overlap_mapping(
            gdf_base, SAFE_BASE_ID, gdf_h3_poly.to_crs("EPSG:3857"), h3_areas, threshold=0.0
        )

        print(f"[Preprocess] Building H3 -> {base_scale} belonging (with orphan recovery)...")
        h3_map = _resolve_parent_mapping(
            gdf_child=gdf_h3_poly, child_id_col='h3_id',
            gdf_base=gdf_base, base_id_col=SAFE_BASE_ID, label="H3->Base"
        )
        mappings['h3_parent'] = h3_map

    # 3. Process all active task scales
    scale_configs = {}
    for t_cfg in Config.TASKS.values():
        t_type = t_cfg['type']
        if t_type in ['street_shp', 'grid_shp']:
            if t_type == 'street_shp':
                s_key = t_cfg.get('scale_key', 'street_normal')
            else:
                s_key = t_cfg.get('scale_key')

            if s_key not in scale_configs:
                scale_configs[s_key] = {'path': t_cfg['path'], 'id_col': t_cfg['id_col'], 'type': t_type}

    for s_key, cfg in scale_configs.items():
        if s_key == base_scale:
            continue

        print(f"[Preprocess] Processing task scale ({s_key}) -> ({base_scale})...")
        gdf_curr = gpd.read_file(cfg['path']).to_crs("EPSG:4326")
        curr_id_col = cfg['id_col']
        gdf_curr[curr_id_col] = gdf_curr[curr_id_col].astype(str)

        mappings['geometric_counts'][s_key] = len(gdf_curr)

        # 3.1 H3 overlap (for feature aggregation, always mapped to H3)
        threshold = Config.GRID_THRESHOLD if cfg['type'] == 'grid_shp' else 0.0
        mappings['overlap'][s_key] = _compute_overlap_mapping(
            gdf_curr, curr_id_col, gdf_h3_poly.to_crs("EPSG:3857"), h3_areas, threshold=threshold
        )

        # 3.2 Parent mapping (for data splitting)
        parent_map = _resolve_parent_mapping(
            gdf_child=gdf_curr, child_id_col=curr_id_col,
            gdf_base=gdf_base, base_id_col=SAFE_BASE_ID, label=f"{s_key}->Base"
        )
        mappings['parent'][s_key] = parent_map

    # 4. Save cache
    if not os.path.exists(Config.MAPPING_CACHE_DIR):
        os.makedirs(Config.MAPPING_CACHE_DIR)
    with open(Config.SPATIAL_MAPPING_CACHE_FILE, 'wb') as f:
        pickle.dump(mappings, f)

    print("[Preprocess] Spatial mappings built.")
    return mappings


# =========================================================
#  2. Utility classes
# =========================================================
class TaskScaler:
    def __init__(self):
        self.stats = {}

    def fit(self, dataset):
        task_data = {t: [] for t in Config.TASKS.keys()}
        for sample in dataset.samples:
            task_data[sample['task_name']].append(sample['label'])
        for t_name, values in task_data.items():
            if not values:
                self.stats[t_name] = {'mean': 0.0, 'std': 1.0}
            else:
                self.stats[t_name] = {'mean': float(np.mean(values)), 'std': float(np.std(values)) + 1e-8}

    def transform(self, labels, task_names):
        return torch.tensor(
            [(l - self.stats[t]['mean']) / self.stats[t]['std'] for l, t in zip(labels, task_names)],
            device=Config.DEVICE, dtype=torch.float32
        )

    def inverse_transform(self, preds, task_names):
        return np.array(
            [p * self.stats[t]['std'] + self.stats[t]['mean']
             for p, t in zip(preds.detach().cpu().numpy(), task_names)]
        )


def get_task_embeddings():
    """Encode scale and task descriptions separately with a frozen BERT.

    Returns:
        (scale_embs, task_type_embs): dicts mapping task name -> [768] tensor.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(Config.BERT_PATH)
        model = AutoModel.from_pretrained(Config.BERT_PATH)
    except Exception as e:
        raise RuntimeError(
            f"[Error] Failed to load BERT model. Check the path.\n"
            f"  Path: {Config.BERT_PATH}\n  Error: {e}"
        )
    model.eval()
    model.to(Config.DEVICE)

    def encode(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(Config.DEVICE)
        return model(**inputs).last_hidden_state[:, 0, :].squeeze(0)

    scale_embs = {}
    task_type_embs = {}

    with torch.no_grad():
        for t_name, t_cfg in Config.TASKS.items():
            s_tag = t_cfg.get('scale_tag', '')
            scale_text = Config.SCALE_MAP.get(s_tag, {}).get('cn', '')
            scale_embs[t_name] = encode(scale_text or t_cfg['prompt_cn'])

            t_tag = t_cfg.get('task_tag', '')
            task_text = Config.TASK_TYPE_MAP.get(t_tag, {}).get('cn', '')
            task_type_embs[t_name] = encode(task_text or t_cfg['prompt_cn'])

    return scale_embs, task_type_embs


def evaluate_metrics(true, pred, task_name="Overall", print_log=True):
    if len(true) == 0:
        return np.nan, np.nan, np.nan
    mae, mse, r2 = mean_absolute_error(true, pred), mean_squared_error(true, pred), r2_score(true, pred)
    if print_log:
        print(f"  > [{task_name}] MAE: {mae:.2f}, RMSE: {np.sqrt(mse):.2f}, R2: {r2:.4f}")
    return mae, np.sqrt(mse), r2


# =========================================================
#  3. Map visualization (multi-scale)
# =========================================================
def visualize_split(train_dataset, val_dataset, output_path, restrict_tasks=None):
    """Visualize the train/validation split across scales."""
    print("Generating dataset split visualization map...")
    try:
        mappings = train_dataset.spatial_mappings
        base_scale = mappings['base_scale']

        train_base_ids = set(train_dataset.target_base_ids)
        test_base_ids = set(val_dataset.target_base_ids)

        all_overlap_scales = sorted(list(mappings['overlap'].keys()))

        if restrict_tasks is not None:
            active_scales = set()
            for t_cfg in restrict_tasks.values():
                t_type = t_cfg['type']
                if t_type == 'street_shp':
                    active_scales.add(t_cfg.get('scale_key', 'street_normal'))
                elif t_type == 'grid_shp':
                    active_scales.add(t_cfg.get('scale_key'))

            other_scales = []
            for s in all_overlap_scales:
                if s == base_scale:
                    continue
                if s in active_scales:
                    other_scales.append(s)
        else:
            other_scales = [s for s in all_overlap_scales if s != base_scale]

        scales_to_plot = []
        if base_scale != 'h3':
            scales_to_plot.append(base_scale)
        scales_to_plot.append('H3')
        scales_to_plot.extend(other_scales)

        n_plots = len(scales_to_plot)
        n_cols = 2
        n_rows = math.ceil(n_plots / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10 * n_rows))

        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        for i in range(n_plots, len(axes)):
            axes[i].axis('off')

        color_train_fill = '#4e79a7'
        color_test_fill = '#2ca02c'
        color_orphan_fill = '#d62728'
        color_nofeat_fill = '#333333'

        alpha_fill = 0.6
        linewidth = 0.3

        gdf_boundary = None
        if os.path.exists(Config.BOUNDARY_PATH):
            try:
                gdf_boundary = gpd.read_file(Config.BOUNDARY_PATH).to_crs("EPSG:4326")
            except Exception:
                pass

        for idx, s_name in enumerate(scales_to_plot):
            ax = axes[idx]
            stats_str = "No Data / Config Missing"

            if gdf_boundary is not None:
                gdf_boundary.plot(ax=ax, color='#f8f8f8', edgecolor='#cccccc', linewidth=1, zorder=0)

            cnt_train = 0
            cnt_test = 0
            cnt_orphan = 0
            cnt_no_feat = 0

            # Case A: H3
            if s_name == 'H3':
                all_h3_ids = mappings.get('valid_h3_ids', [])
                parent_map = mappings['h3_parent']

                h3_train = []
                h3_test = []
                h3_orphan = []

                for hid in all_h3_ids:
                    pid = parent_map.get(hid)
                    if pid in train_base_ids:
                        h3_train.append(hid)
                    elif pid in test_base_ids:
                        h3_test.append(hid)
                    else:
                        h3_orphan.append(hid)

                cnt_train = len(h3_train)
                cnt_test = len(h3_test)
                cnt_orphan = len(h3_orphan)

                def plot_h3_list(h3_list, fill_c, label):
                    if not h3_list:
                        return
                    try:
                        polys = [Polygon(h3.h3_to_geo_boundary(hid, geo_json=True)) for hid in h3_list]
                        gdf_h3 = gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326")
                        if not gdf_h3.empty:
                            gdf_h3.plot(ax=ax, color=fill_c, edgecolor=fill_c, alpha=alpha_fill, linewidth=0.1, zorder=1, label=label)
                    except Exception:
                        pass

                plot_h3_list(h3_train, color_train_fill, 'Train')
                plot_h3_list(h3_test, color_test_fill, 'Test')
                plot_h3_list(h3_orphan, color_orphan_fill, 'Orphan')

                total = len(all_h3_ids)
                stats_str = (f"Total: {total} | Train: {cnt_train} | Test: {cnt_test}\n"
                             f"Orphan: {cnt_orphan} | No Feat: {cnt_no_feat}")

            # Case B: polygon
            else:
                target_cfg = None
                search_source = restrict_tasks if restrict_tasks is not None else Config.TASKS

                def find_cfg(source, scale):
                    if scale == base_scale:
                        for t in source.values():
                            t_key = t.get('scale_key', 'street_normal') if t['type'] == 'street_shp' else t.get('scale_key')
                            if t['type'] == 'street_shp' and t_key == scale:
                                return t
                    else:
                        for t in source.values():
                            if t.get('scale_key') == scale:
                                return t
                    return None

                target_cfg = find_cfg(search_source, s_name)

                if target_cfg is None:
                    target_cfg = find_cfg(Config.TASKS, s_name)

                if target_cfg is None and s_name in Config.SCALE_DEFINITIONS:
                    target_cfg = Config.SCALE_DEFINITIONS[s_name]

                if target_cfg and os.path.exists(target_cfg['path']):
                    gdf = gpd.read_file(target_cfg['path']).to_crs("EPSG:4326")
                    id_col = target_cfg['id_col']
                    gdf[id_col] = gdf[id_col].astype(str)

                    splits = []
                    parent_map = mappings['parent'].get(s_name, {}) if s_name != base_scale else None
                    overlap_map = mappings['overlap'].get(s_name, {})

                    for _, row in gdf.iterrows():
                        pid = row[id_col]
                        has_feature = pid in overlap_map

                        owner_base_id = None
                        if s_name == base_scale:
                            owner_base_id = pid
                        else:
                            owner_base_id = parent_map.get(pid)

                        if has_feature:
                            if owner_base_id in train_base_ids:
                                splits.append('train')
                            elif owner_base_id in test_base_ids:
                                splits.append('test')
                            else:
                                splits.append('orphan')
                        else:
                            splits.append('no_feature')

                    gdf['split_status'] = splits

                    gdf_train = gdf[gdf['split_status'] == 'train']
                    gdf_test = gdf[gdf['split_status'] == 'test']
                    gdf_orphan = gdf[gdf['split_status'] == 'orphan']
                    gdf_no_feat = gdf[gdf['split_status'] == 'no_feature']

                    if not gdf_train.empty:
                        gdf_train.plot(ax=ax, color=color_train_fill, edgecolor='white', linewidth=linewidth, alpha=alpha_fill, zorder=2)
                    if not gdf_test.empty:
                        gdf_test.plot(ax=ax, color=color_test_fill, edgecolor='white', linewidth=linewidth, alpha=alpha_fill, zorder=2)
                    if not gdf_orphan.empty:
                        gdf_orphan.plot(ax=ax, color=color_orphan_fill, edgecolor='white', linewidth=linewidth, alpha=0.6, zorder=1)
                    if not gdf_no_feat.empty:
                        gdf_no_feat.plot(ax=ax, color=color_nofeat_fill, edgecolor='white', linewidth=linewidth, alpha=0.3, zorder=1)

                    cnt_train = len(gdf_train)
                    cnt_test = len(gdf_test)
                    cnt_orphan = len(gdf_orphan)
                    cnt_no_feat = len(gdf_no_feat)

                    total = cnt_train + cnt_test + cnt_orphan + cnt_no_feat
                    stats_str = (f"Total: {total} | Train: {cnt_train} | Test: {cnt_test}\n"
                                 f"Orphan: {cnt_orphan} | No Feat: {cnt_no_feat}")

            ax.set_title(f"Scale: {s_name}", fontsize=16, fontweight='bold', pad=15)
            ax.text(0.5, -0.02, stats_str, transform=ax.transAxes, ha='center', va='top', fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color_train_fill, label='Train Set'),
            Patch(facecolor=color_test_fill, label='Test Set'),
            Patch(facecolor=color_orphan_fill, alpha=0.6, label='Orphan (Valid Feat, No Parent)'),
            Patch(facecolor=color_nofeat_fill, alpha=0.3, label='No Feature (Invalid H3)')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01), fontsize=14)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12, hspace=0.35)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization map saved to: {output_path}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Map visualization failed: {e}")


def visualize_k_fold_partition(all_base_ids, kf, output_path):
    """Visualize the global K-Fold spatial partition (adapts to Config.BASE_SCALE)."""
    print(f"Generating global K-Fold partition map (total items: {len(all_base_ids)})...")
    try:
        base_scale = Config.BASE_SCALE

        gdf_plot = None
        id_col = 'temp_id'

        # Branch A: H3 base (build hexagons)
        if base_scale == 'h3':
            print("  [Info] H3 mode: generating hexagon geometries...")

            geoms = []
            valid_ids = []

            for hid in tqdm(all_base_ids, desc="Generating H3 Polygons", leave=False):
                try:
                    hid_str = str(hid)
                    poly = Polygon(h3.h3_to_geo_boundary(hid_str, geo_json=True))
                    geoms.append(poly)
                    valid_ids.append(hid_str)
                except Exception:
                    continue

            if not valid_ids:
                print("  [Error] H3 geometry generation failed, list is empty.")
                return

            gdf_plot = gpd.GeoDataFrame({id_col: valid_ids}, geometry=geoms, crs="EPSG:4326")

        # Branch B: polygon base (load SHP)
        else:
            try:
                base_info = _get_base_scale_geometry(base_scale)
            except Exception as e:
                print(f"  [Warning] Could not resolve base scale [{base_scale}] geometry: {e}")
                return

            if base_info['type'] != 'shp':
                print("  [Skip] Unsupported geometry type.")
                return

            shp_path = base_info['path']
            real_id_col = base_info['id_col']

            if not os.path.exists(shp_path):
                print(f"  [Error] SHP file not found: {shp_path}")
                return

            gdf_plot = gpd.read_file(shp_path)
            gdf_plot = gdf_plot.rename(columns={real_id_col: id_col})
            gdf_plot[id_col] = gdf_plot[id_col].astype(str)

        # Map base id -> fold id
        base_to_fold = {}
        for fold, (_, test_idx) in enumerate(kf.split(all_base_ids)):
            current_ids = all_base_ids[test_idx]
            for bid in current_ids:
                base_to_fold[str(bid)] = fold

        gdf_plot['fold_id'] = gdf_plot[id_col].map(base_to_fold)
        gdf_final = gdf_plot.dropna(subset=['fold_id']).copy()

        if gdf_final.empty:
            print("  [Warning] Mapped data is empty. Check id consistency:")
            print(f"   - all_base_ids sample: {all_base_ids[0] if len(all_base_ids) > 0 else 'Empty'}")
            print(f"   - GDF id sample: {gdf_plot[id_col].iloc[0] if not gdf_plot.empty else 'Empty'}")
            return

        fig, ax = plt.subplots(figsize=(12, 12))

        if os.path.exists(Config.BOUNDARY_PATH):
            try:
                boundary = gpd.read_file(Config.BOUNDARY_PATH).to_crs("EPSG:4326")
                boundary.plot(ax=ax, facecolor='none', edgecolor='#333333', linewidth=1.5, zorder=10, alpha=0.5)
            except Exception:
                pass

        lw = 0.2 if base_scale == 'h3' else 0.1

        gdf_final.plot(column='fold_id', ax=ax, categorical=True, legend=True,
                       cmap='tab10', edgecolor='white', linewidth=lw, alpha=0.9,
                       legend_kwds={'title': 'Fold ID', 'loc': 'lower right'})

        plt.title(f"K-Fold Spatial Partition\nBase: {base_scale}, Items: {len(gdf_final)}, K={kf.get_n_splits()}", fontsize=15)
        plt.axis('off')
        plt.tight_layout()

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Global partition map saved: {output_path}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [Error] Global partition map failed: {e}")


# =========================================================
#  4. Training curves (N x 3 subplots across scales)
# =========================================================
def plot_training_curves(history_df, output_path):
    print(f"Plotting training curves... (columns: {list(history_df.columns)[:5]})")
    try:
        epochs = history_df['epoch']

        # Part 1: N x 3 multi-scale metric plots
        scales = set()
        for col in history_df.columns:
            if '_all_r2' in col and 'global' not in col:
                scale = col.replace('_all_r2', '')
                scales.add(scale)

        sorted_scales = sorted(list(scales))

        def sort_key(s):
            if s == 'h3':
                return 0
            if 'street_normal' in s:
                return 1
            if s.startswith('L') and s[1:].isdigit():
                return 2
            if 'grid' in s or 'km' in s:
                return 3
            return 4

        sorted_scales.sort(key=sort_key)

        if sorted_scales:
            n_rows = len(sorted_scales)
            fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows), squeeze=False)
            metrics_cfg = [('r2', 'R2 Score'), ('rmse', 'RMSE'), ('mae', 'MAE')]

            for row_idx, scale in enumerate(sorted_scales):
                scale_title = scale.upper()
                if scale == 'h3':
                    scale_title = "H3 Grid"
                elif scale == 'street_normal':
                    scale_title = "Street (Normal)"
                elif scale.startswith('grid_'):
                    scale_title = f"Grid ({scale.replace('grid_', '')})"

                for col_idx, (m_key, m_name) in enumerate(metrics_cfg):
                    ax = axes[row_idx, col_idx]

                    avg_col = f"{scale}_all_{m_key}"
                    if avg_col in history_df.columns:
                        ax.plot(epochs, history_df[avg_col], label='ALL (Avg)',
                                color='black', linewidth=2.5, linestyle='-', alpha=0.6, zorder=10)

                    for t_name, t_cfg in Config.TASKS.items():
                        t_scale = 'unknown'
                        if t_cfg['type'] == 'h3_csv':
                            t_scale = 'h3'
                        elif t_cfg['type'] in ['street_shp', 'grid_shp']:
                            t_scale = t_cfg.get('scale_key', 'street_normal')

                        if t_scale == scale:
                            col_name = f"{t_name}_{m_key}"
                            if col_name in history_df.columns:
                                ax.plot(epochs, history_df[col_name], label=t_name,
                                        linewidth=1.5, linestyle='--', alpha=0.8)

                    ax.set_title(f"{scale_title} - {m_name}", fontsize=12, fontweight='bold')
                    ax.grid(True, linestyle='--', alpha=0.6)
                    ax.set_xlabel("Epoch")
                    if m_key == 'r2':
                        ax.set_ylim(-0.5, 1.05)
                    ax.legend(fontsize=8, loc='best')

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Metric curves saved to: {output_path}")
        else:
            print("No valid scale data found, skipping metric plots.")

        # Part 2: loss curve
        loss_output_path = os.path.join(os.path.dirname(output_path), "training_loss.png")
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history_df['train_loss'], label='Total Train Loss',
                 color='black', linewidth=2, marker='o', markersize=4)
        plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
        plt.ylabel('MSE Loss')
        plt.xlabel('Epoch')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(loss_output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training loss curve saved to: {loss_output_path}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Plotting failed: {e}")


def init_task_prompts(tasks_dict, scale_map, task_type_map):
    """Build prompt text for each task in-place by combining scale and task descriptions.

    Args:
        tasks_dict: task config dict (e.g. Config.TASKS)
        scale_map: scale description map (Config.SCALE_MAP)
        task_type_map: task description map (Config.TASK_TYPE_MAP)
    """
    for task_key, task_cfg in tasks_dict.items():
        s_tag = task_cfg.get('scale_tag')
        t_tag = task_cfg.get('task_tag')

        if not s_tag or not t_tag:
            continue

        if s_tag not in scale_map or t_tag not in task_type_map:
            print(f"[Utils Warning] Task '{task_key}' has invalid tags: {s_tag}, {t_tag}")
            continue

        scale_info = scale_map[s_tag]
        task_info = task_type_map[t_tag]

        task_cfg['prompt_cn'] = f"{scale_info['cn']}，{task_info['cn']}"
        task_cfg['prompt'] = f"{scale_info['en']}, {task_info['en']}"


def filter_active_tasks(config_class):
    """Filter Config.TASKS down to Config.ACTIVE_TASKS, in place."""
    if not hasattr(config_class, 'ACTIVE_TASKS') or not config_class.ACTIVE_TASKS:
        print("Warning: Config.ACTIVE_TASKS is empty or not found. Using all TASKS.")
        return

    filtered_tasks = {}
    for task_key in config_class.ACTIVE_TASKS:
        if task_key in config_class.TASKS:
            filtered_tasks[task_key] = config_class.TASKS[task_key]
        else:
            print(f"Warning: active task '{task_key}' not defined in Config.TASKS, ignored.")

    config_class.TASKS = filtered_tasks

    loaded_keys = list(config_class.TASKS.keys())
    print(f"Successfully loaded {len(config_class.TASKS)} active tasks: {loaded_keys}")


# =========================================================
#  5. Results -> Markdown tables
# =========================================================

def results_to_md_tables(results_summary):
    """Convert a results summary list into four Markdown tables.

    Table 1: per-task K-fold average performance
    Table 2: per-scale aggregated performance
    Table 3: global aggregated performance
    Table 4: per-task-family average performance
    """
    if not results_summary:
        return ""

    df = pd.DataFrame(results_summary)
    df.columns = [c.strip() for c in df.columns]

    for c in ['Samples', 'Mean', 'Std', 'Valid', 'Orphan', 'No Feat', 'MAE', 'RMSE', 'R2']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    INT_COLS = {'Samples', 'Valid', 'Orphan', 'No Feat'}

    def fmt(col, val):
        if pd.isna(val):
            return ''
        if col in INT_COLS:
            return str(int(round(val)))
        return f'{val:.4f}'

    def md_table(cols, rows_iter):
        lines = []
        lines.append('| ' + ' | '.join(cols) + ' |')
        lines.append('|' + '|'.join([':---'] * len(cols)) + '|')
        for row in rows_iter:
            lines.append('| ' + ' | '.join(str(row.get(c, '')) for c in cols) + ' |')
        return '\n'.join(lines)

    if 'Type' in df.columns:
        task_df = df[df['Type'].isin(['grid_shp', 'street_shp', 'h3_csv'])].copy()
        global_df = df[df['Type'] == 'Global_Agg'].copy()
    else:
        task_df = df.copy()
        global_df = pd.DataFrame()

    # Table 1
    T1_COLS = ['Task', 'R2', 'Samples', 'Mean', 'Std', 'Valid', 'Orphan', 'No Feat', 'MAE', 'RMSE']

    def t1_rows():
        for _, r in task_df.iterrows():
            yield {c: (r[c] if c == 'Task' else fmt(c, r[c])) for c in T1_COLS}

    table1 = md_table(T1_COLS, t1_rows())

    # Table 2
    METRIC_COLS = ['Samples', 'Mean', 'Std', 'Valid', 'Orphan', 'No Feat', 'MAE', 'RMSE', 'R2']
    T2_COLS = ['Task', 'R2', 'Samples', 'Mean', 'Std', 'Valid', 'Orphan', 'No Feat', 'MAE', 'RMSE']

    if not task_df.empty and 'Scale' in task_df.columns:
        grouped = task_df.groupby('Scale', sort=False)[METRIC_COLS].mean().reset_index()
    else:
        grouped = pd.DataFrame(columns=['Scale'] + METRIC_COLS)

    def t2_rows():
        for _, r in grouped.iterrows():
            row = {'Task': f"[Agg] {r['Scale']}"}
            for c in T2_COLS:
                if c == 'Task':
                    continue
                row[c] = fmt(c, r[c])
            yield row

    table2 = md_table(T2_COLS, t2_rows())

    # Table 3
    T3_COLS = ['Task', 'R2', 'Samples', 'Mean', 'Std', 'Valid', 'Orphan', 'No Feat', 'MAE', 'RMSE', 'Train_Loss']

    def t3_rows():
        if not global_df.empty:
            for _, r in global_df.iterrows():
                row = {}
                for c in T3_COLS:
                    if c == 'Task':
                        row[c] = '[Global Summary]'
                    else:
                        row[c] = fmt(c, r[c]) if c in r.index else ''
                yield row
        else:
            row = {'Task': '[Global Summary]'}
            for c in T3_COLS:
                if c == 'Task':
                    continue
                row[c] = fmt(c, task_df[c].mean()) if c in task_df.columns else ''
            yield row

    table3 = md_table(T3_COLS, t3_rows())

    # Table 4
    if not task_df.empty:
        task_df['base_task'] = task_df['Task'].str.split('_').str[0]
        res = task_df.groupby('base_task')[['R2', 'Mean', 'Std', 'MAE', 'RMSE']].mean().reset_index()
        res = res.rename(columns={'base_task': 'Task'})
    else:
        res = pd.DataFrame(columns=['Task', 'R2', 'Mean', 'Std', 'MAE', 'RMSE'])

    T4_COLS = ['Task', 'R2', 'Mean', 'Std', 'MAE', 'RMSE']

    def t4_rows():
        for _, r in res.iterrows():
            yield {c: (r[c] if c == 'Task' else f'{r[c]:.4f}') for c in T4_COLS}

    table4 = md_table(T4_COLS, t4_rows())

    return (
        "### 1. Per-task K-fold average performance\n"
        + table1
        + "\n\n### 2. Per-scale aggregated performance\n"
        + table2
        + "\n\n### 3. Global aggregated performance\n"
        + table3
        + "\n\n### 4. Per-task-family average performance\n"
        + table4
    )
