import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import pickle
from collections import defaultdict
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from tqdm import tqdm

from config import Config
from model import UrbanPromptModel
from dataset import UrbanDataset, urban_collate_fn
from utils import (
    TaskScaler,
    get_task_embeddings,
    build_spatial_mappings,
    visualize_split,
    plot_training_curves,
    visualize_k_fold_partition,
    evaluate_metrics,
    init_task_prompts,
    filter_active_tasks,
    load_config_from_json,
    results_to_md_tables
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train():
    # ===========================================
    # 0. Setup and spatial mappings
    # ===========================================
    load_config_from_json(Config)
    filter_active_tasks(Config)
    init_task_prompts(Config.TASKS, Config.SCALE_MAP, Config.TASK_TYPE_MAP)

    print("=== Building spatial mappings ===")
    spatial_mappings = build_spatial_mappings()

    base_scale = spatial_mappings.get('base_scale', Config.BASE_SCALE)
    print(f"\n[K-Fold] Base scale for splitting: {base_scale}")

    all_base_ids = []
    if base_scale == 'h3':
        all_base_ids = spatial_mappings.get('valid_h3_ids', [])
    else:
        base_overlap_map = spatial_mappings['overlap'].get(base_scale, {})
        all_base_ids = list(base_overlap_map.keys())

    all_base_ids.sort()
    all_base_ids = np.array(all_base_ids)
    print(f"[K-Fold] Valid base units: {len(all_base_ids)}, running {Config.K_FOLDS}-fold cross-validation.")

    Config.OUTPUT_DIR = os.path.join(Config.ROOT_DIR, "prompt_learning/output/train", Config.CITYNAME, datetime.now().strftime("%Y%m%d_%H%M%S"))
    if os.path.exists(Config.OUTPUT_DIR):
        print(f"Output dir {Config.OUTPUT_DIR} exists, clearing...")
        shutil.rmtree(Config.OUTPUT_DIR)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    shutil.copy(os.path.join(Config.ROOT_DIR, "prompt_learning/config.py"), Config.OUTPUT_DIR)
    print(f"Output dir ready: {Config.OUTPUT_DIR}")
    print(f"Config backed up to: {Config.OUTPUT_DIR}")

    print("\nGenerating task embeddings (BERT)...")
    scale_emb_dict, task_type_emb_dict = get_task_embeddings()
    device = Config.DEVICE

    # =========================================================
    # 0.5 Global per-task statistics
    # =========================================================
    print("\n[Step 0.5] Computing global per-task statistics...")
    task_global_stats = {}
    h3_to_idx_dummy = {hid: i for i, hid in enumerate(spatial_mappings['valid_h3_ids'])}

    for t_name, t_cfg in Config.TASKS.items():
        try:
            df_stats = None
            if t_cfg['type'] == 'h3_csv':
                df_stats = pd.read_csv(t_cfg['path'])
            else:
                df_stats = gpd.read_file(t_cfg['path'])

            id_col = t_cfg['id_col']
            df_stats[id_col] = df_stats[id_col].astype(str)

            target_col = t_cfg['target_col']
            valid_rows = df_stats[~df_stats[target_col].isna()]

            raw_vals = valid_rows[target_col].values
            gt_mean = float(np.mean(raw_vals))
            gt_std = float(np.std(raw_vals)) + 1e-8

            cnt_valid, cnt_orphan, cnt_nofeat = 0, 0, 0

            scale_key = t_cfg.get('scale_key', 'street_normal') if t_cfg['type'] == 'street_shp' else t_cfg.get('scale_key')

            overlap_map = {}
            parent_map = {}
            if t_cfg['type'] == 'h3_csv':
                overlap_map = {hid: True for hid in h3_to_idx_dummy}
                parent_map = spatial_mappings['h3_parent']
            else:
                overlap_map = spatial_mappings['overlap'].get(scale_key, {})
                if scale_key != base_scale:
                    parent_map = spatial_mappings['parent'].get(scale_key, {})

            for _, row in valid_rows.iterrows():
                sid = row[id_col]
                has_feat = False
                if t_cfg['type'] == 'h3_csv':
                    has_feat = (sid in h3_to_idx_dummy)
                else:
                    has_feat = (sid in overlap_map)

                if not has_feat:
                    cnt_nofeat += 1
                    continue

                has_parent = False
                if scale_key == base_scale:
                    has_parent = True
                else:
                    has_parent = (parent_map.get(sid) is not None)

                if has_parent:
                    cnt_valid += 1
                else:
                    cnt_orphan += 1

            task_global_stats[t_name] = {
                'Samples': len(valid_rows), 'Mean': gt_mean, 'Std': gt_std,
                'Valid': cnt_valid, 'Orphan': cnt_orphan, 'No Feat': cnt_nofeat
            }
        except Exception as e:
            print(f"  [Warning] Stat calc failed for {t_name}: {e}")
            task_global_stats[t_name] = {'Samples': 0, 'Mean': 0, 'Std': 0, 'Valid': 0, 'Orphan': 0, 'No Feat': 0}

    k_fold_best_metrics = []
    all_folds_history = []
    kf = KFold(n_splits=Config.K_FOLDS, shuffle=True, random_state=Config.SPLIT_SEED)

    # Visualize the global K-Fold partition
    map_output_path = os.path.join(Config.OUTPUT_DIR, "k_fold_partition_map.png")
    visualize_k_fold_partition(all_base_ids, kf, map_output_path)

    # ===========================================
    # 1. K-Fold loop
    # ===========================================
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_base_ids)):
        print(f"\n{'='*60}")
        print(f"  Fold {fold}/{Config.K_FOLDS-1}")
        print(f"{'='*60}")

        fold_dir = os.path.join(Config.OUTPUT_DIR, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_base_ids = all_base_ids[train_idx]
        val_base_ids = all_base_ids[val_idx]

        # Funnel statistics
        print(f"\n>>> [Fold {fold}] Split statistics:")
        header = f"    {'Scale':<12} | {'Total(SHP)':<10} | {'Train':<8} | {'Test':<8} | {'Orphan':<8} | {'No Feat':<8}"
        print(header)
        print("    " + "-" * 80)

        geo_counts = spatial_mappings.get('geometric_counts', {})
        base_total = geo_counts.get(base_scale, 0)
        base_active = len(train_base_ids) + len(val_base_ids)
        base_orphan = len(all_base_ids) - base_active
        base_no_feat = base_total - len(all_base_ids)
        print(f"    {base_scale:<12} | {base_total:<10} | {len(train_base_ids):<8} | {len(val_base_ids):<8} | {base_orphan:<8} | {base_no_feat:<8}")

        if base_scale != 'h3':
            all_h3_ids = spatial_mappings.get('valid_h3_ids', [])
            h3_parent_map = spatial_mappings['h3_parent']
            train_set = set(train_base_ids)
            val_set = set(val_base_ids)
            h3_train = sum(1 for hid in all_h3_ids if h3_parent_map.get(hid) in train_set)
            h3_test = sum(1 for hid in all_h3_ids if h3_parent_map.get(hid) in val_set)
            h3_active = h3_train + h3_test
            h3_orphan = len(all_h3_ids) - h3_active
            print(f"    {'H3':<12} | {len(all_h3_ids):<10} | {h3_train:<8} | {h3_test:<8} | {h3_orphan:<8} | {0:<8}")

        other_scales = sorted([s for s in spatial_mappings['overlap'].keys() if s != base_scale])
        for s_key in other_scales:
            s_total = geo_counts.get(s_key, 0)
            candidate_ids = list(spatial_mappings['overlap'][s_key].keys())
            parent_map = spatial_mappings['parent'].get(s_key, {})
            train_set = set(train_base_ids)
            val_set = set(val_base_ids)
            s_train, s_test = 0, 0
            for vid in candidate_ids:
                pid = parent_map.get(vid)
                if pid in train_set:
                    s_train += 1
                elif pid in val_set:
                    s_test += 1
            s_active = s_train + s_test
            s_orphan = len(candidate_ids) - s_active
            s_no_feat = s_total - len(candidate_ids)
            print(f"    {s_key:<12} | {s_total:<10} | {s_train:<8} | {s_test:<8} | {s_orphan:<8} | {s_no_feat:<8}")
        print("    " + "-" * 80 + "\n")

        # Init datasets
        print("--> Initializing datasets...")
        train_dataset = UrbanDataset(mode='train', spatial_mappings=spatial_mappings, specific_base_ids=train_base_ids)
        val_dataset = UrbanDataset(mode='test', spatial_mappings=spatial_mappings, specific_base_ids=val_base_ids)

        # Visualize split
        print("--> Plotting spatial split map...")
        map_output_path = os.path.join(fold_dir, "split_visualization.png")
        visualize_split(train_dataset, val_dataset, map_output_path)

        # DataLoaders
        train_weights = train_dataset.get_weights()
        if len(train_weights) > 0:
            sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, sampler=sampler, collate_fn=urban_collate_fn, num_workers=4)
        else:
            print("Warning: training set is empty!")
            train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=urban_collate_fn, num_workers=4)

        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=urban_collate_fn, num_workers=4)

        # Init model
        print("--> Initializing model...")
        model = UrbanPromptModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=Config.LR_FACTOR, patience=Config.LR_PATIENCE)
        criterion = nn.MSELoss()

        scaler = TaskScaler()
        scaler.fit(train_dataset)

        # Training loop
        best_global_r2 = -float('inf')
        best_epoch_data = {}
        no_improve_epochs = 0
        patience = 10
        fold_history = []

        for epoch in range(Config.EPOCHS):
            model.train()
            total_loss = 0
            train_steps = 0

            progress_bar = tqdm(train_loader, desc=f"[Fold {fold}] Epoch {epoch+1}/{Config.EPOCHS}", leave=False)
            for batch in progress_bar:
                feats = batch['feats'].to(device)
                weights = batch['weights'].to(device)
                masks = batch['masks'].to(device)
                labels = batch['labels'].to(device)
                task_names = batch['task_names']

                targets_norm = scaler.transform(labels, task_names)
                current_scale_embs = torch.stack([scale_emb_dict[t] for t in task_names]).to(device)
                current_task_type_embs = torch.stack([task_type_emb_dict[t] for t in task_names]).to(device)

                preds_norm = model(feats, weights, masks, current_scale_embs, current_task_type_embs)
                loss = criterion(preds_norm, targets_norm)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                train_steps += 1
                progress_bar.set_postfix({'loss': loss.item()})

            avg_train_loss = total_loss / train_steps if train_steps > 0 else 0

            model.eval()
            val_results = {t: {'true': [], 'pred': []} for t in Config.TASKS.keys()}

            with torch.no_grad():
                for batch in val_loader:
                    feats = batch['feats'].to(device)
                    weights = batch['weights'].to(device)
                    masks = batch['masks'].to(device)
                    labels = batch['labels']
                    task_names = batch['task_names']

                    current_scale_embs = torch.stack([scale_emb_dict[t] for t in task_names]).to(device)
                    current_task_type_embs = torch.stack([task_type_emb_dict[t] for t in task_names]).to(device)
                    preds_norm = model(feats, weights, masks, current_scale_embs, current_task_type_embs)
                    preds_real = scaler.inverse_transform(preds_norm, task_names)
                    if isinstance(preds_real, torch.Tensor):
                        preds_real = preds_real.detach().cpu().numpy()
                    raw_labels = labels.numpy()

                    for i, t_name in enumerate(task_names):
                        val_results[t_name]['true'].append(raw_labels[i])
                        val = preds_real[i]
                        if isinstance(val, torch.Tensor):
                            val = val.item()
                        val_results[t_name]['pred'].append(val)

            # Metrics
            current_metrics = {'fold': fold, 'epoch': epoch + 1, 'train_loss': avg_train_loss}
            scale_metrics_collector = defaultdict(lambda: defaultdict(list))
            all_task_r2_list = []

            for t_name, data in val_results.items():
                true_vals = np.array(data['true'])
                pred_vals = np.array(data['pred'])
                if len(true_vals) == 0:
                    continue

                t_mae, t_rmse, t_r2 = evaluate_metrics(true_vals, pred_vals, task_name=t_name, print_log=False)

                current_metrics[f"{t_name}_mae"] = t_mae
                current_metrics[f"{t_name}_rmse"] = t_rmse
                current_metrics[f"{t_name}_r2"] = t_r2
                all_task_r2_list.append(t_r2)

                task_cfg = Config.TASKS[t_name]
                t_type = task_cfg['type']
                scale_group = "unknown"
                if t_type == 'h3_csv':
                    scale_group = "h3"
                elif t_type == 'street_shp':
                    scale_group = task_cfg.get('scale_key', 'street_normal')
                elif t_type == 'grid_shp':
                    scale_group = task_cfg['scale_key']

                scale_metrics_collector[scale_group]['mae'].append(t_mae)
                scale_metrics_collector[scale_group]['rmse'].append(t_rmse)
                scale_metrics_collector[scale_group]['r2'].append(t_r2)

            log_strings = []
            all_groups = sorted(scale_metrics_collector.keys())

            for group in all_groups:
                metrics = scale_metrics_collector[group]
                if not metrics['r2']:
                    continue
                avg_r2 = np.mean(metrics['r2'])
                current_metrics[f"{group}_all_mae"] = np.mean(metrics['mae'])
                current_metrics[f"{group}_all_rmse"] = np.mean(metrics['rmse'])
                current_metrics[f"{group}_all_r2"] = avg_r2
                log_strings.append(f"{group}: R2={avg_r2:.4f}")

            if len(all_task_r2_list) > 0:
                current_global_r2 = np.mean(all_task_r2_list)
            else:
                current_global_r2 = 0.0

            current_metrics["global_all_r2"] = current_global_r2
            print(f"  > Epoch {epoch+1} | Global R2: {current_global_r2:.4f} | " + " | ".join(log_strings))

            fold_history.append(current_metrics)
            all_folds_history.append(current_metrics.copy())

            scheduler.step(current_global_r2)

            if current_global_r2 > best_global_r2:
                best_global_r2 = current_global_r2
                best_epoch_data = current_metrics.copy()
                no_improve_epochs = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'scaler_stats': scaler.stats,
                    'base_scale': base_scale,
                }, os.path.join(fold_dir, "best_model.pth"))
                print(f"    New best generalization (Global R2: {best_global_r2:.4f}), model saved.")
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= patience:
                print("  > Early stopping triggered.")
                break

        # End of fold
        fold_df = pd.DataFrame(fold_history)
        fold_df.to_csv(os.path.join(fold_dir, "training_metrics.csv"), index=False)
        plot_training_curves(fold_df, os.path.join(fold_dir, "training_curves.png"))

        if best_epoch_data:
            k_fold_best_metrics.append(best_epoch_data)
        elif fold_history:
            k_fold_best_metrics.append(fold_history[-1])

    # -----------------------------------------------------
    # Report generation
    # -----------------------------------------------------
    print("\n=================================================")
    print("   Training K-Fold Summary (Avg of Best Epochs)")
    print("=================================================")

    if not k_fold_best_metrics:
        print("No valid training results.")
        return

    # 1. Task-level averages
    task_rows = []
    avg_train_loss = np.mean([log.get('train_loss', 0) for log in k_fold_best_metrics])

    for t_name, t_cfg in Config.TASKS.items():
        r2s = [log.get(f"{t_name}_r2", 0) for log in k_fold_best_metrics if f"{t_name}_r2" in log]
        maes = [log.get(f"{t_name}_mae", 0) for log in k_fold_best_metrics if f"{t_name}_mae" in log]
        rmses = [log.get(f"{t_name}_rmse", 0) for log in k_fold_best_metrics if f"{t_name}_rmse" in log]

        if not r2s:
            continue

        scale = t_cfg.get('scale_key', 'H3' if t_cfg['type'] == 'h3_csv' else 'Street')
        stats = task_global_stats.get(t_name, {'Samples': 0, 'Mean': 0, 'Std': 0, 'Valid': 0, 'Orphan': 0, 'No Feat': 0})

        task_rows.append({
            'Task': t_name,
            'Type': t_cfg['type'],
            'Scale': scale,
            'Samples': stats['Samples'],
            'Mean': stats['Mean'],
            'Std': stats['Std'],
            'Valid': stats['Valid'],
            'Orphan': stats['Orphan'],
            'No Feat': stats['No Feat'],
            'MAE': np.mean(maes),
            'RMSE': np.mean(rmses),
            'R2': np.mean(r2s),
            'Train_Loss': None
        })

    df_tasks = pd.DataFrame(task_rows)

    # 2. Scale-level aggregation
    scale_rows = []
    if not df_tasks.empty:
        for scale, group in df_tasks.groupby('Scale'):
            scale_rows.append({
                'Task': f"[Agg] {scale}",
                'Type': 'Scale_Agg',
                'Scale': scale,
                'Samples': group['Samples'].sum(),
                'Mean': group['Mean'].mean(),
                'Std': group['Std'].mean(),
                'Valid': group['Valid'].sum(),
                'Orphan': group['Orphan'].sum(),
                'No Feat': group['No Feat'].sum(),
                'MAE': group['MAE'].mean(),
                'RMSE': group['RMSE'].mean(),
                'R2': group['R2'].mean(),
                'Train_Loss': None
            })

    # 3. Global aggregation
    global_row = {}
    if not df_tasks.empty:
        global_row = {
            'Task': "[Global Summary]",
            'Type': 'Global_Agg',
            'Scale': 'Global',
            'Samples': df_tasks['Samples'].sum(),
            'Mean': df_tasks['Mean'].mean(),
            'Std': df_tasks['Std'].mean(),
            'Valid': df_tasks['Valid'].sum(),
            'Orphan': df_tasks['Orphan'].sum(),
            'No Feat': df_tasks['No Feat'].sum(),
            'MAE': df_tasks['MAE'].mean(),
            'RMSE': df_tasks['RMSE'].mean(),
            'R2': df_tasks['R2'].mean(),
            'Train_Loss': avg_train_loss
        }

    # 4. Console output
    display_cols = ['Task', 'Type', 'Scale', 'Samples', 'Mean', 'Std', 'Valid', 'Orphan', 'No Feat', 'MAE', 'RMSE', 'R2']

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    float_fmt = lambda x: "{:.4f}".format(x)

    print("\n[1. Per-task K-fold average performance]")
    if not df_tasks.empty:
        df_tasks_disp = df_tasks.sort_values(by=['Scale', 'Task'])[display_cols]
        print(df_tasks_disp.to_string(index=False, float_format=float_fmt))

    print("\n[2. Per-scale aggregated K-fold average performance]")
    if scale_rows:
        df_scales = pd.DataFrame(scale_rows)
        df_scales_disp = df_scales.sort_values(by='Scale')[display_cols]
        print(df_scales_disp.to_string(index=False, float_format=float_fmt))

    print("\n[3. Global aggregated K-fold average performance]")
    if global_row:
        global_disp_cols = display_cols + ['Train_Loss']
        df_global = pd.DataFrame([global_row])[global_disp_cols]
        print(df_global.to_string(index=False, float_format=float_fmt))

    # 5. Save CSV
    all_rows = task_rows + scale_rows + ([global_row] if global_row else [])
    df_final = pd.DataFrame(all_rows)

    cols = ['Task', 'Type', 'Scale', 'Samples', 'Mean', 'Std', 'Valid', 'Orphan', 'No Feat', 'MAE', 'RMSE', 'R2', 'Train_Loss']
    df_final = df_final[cols]

    csv_path = os.path.join(Config.OUTPUT_DIR, "train_k_fold_summary.csv")
    df_final.to_csv(csv_path, index=False)
    print(f"\nFull summary report saved: {csv_path}")

    # Markdown tables
    md_txt = results_to_md_tables(all_rows)
    if md_txt:
        md_path = os.path.join(Config.OUTPUT_DIR, "train_metrics.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_txt)
        print(f"Metric tables saved: {md_path}")

    # Epoch summary
    if len(all_folds_history) > 0:
        all_df = pd.DataFrame(all_folds_history)
        summary_df = all_df.groupby('epoch').mean(numeric_only=True).reset_index()
        if 'fold' in summary_df.columns:
            summary_df = summary_df.drop(columns=['fold'])
        summary_df.to_csv(os.path.join(Config.OUTPUT_DIR, "k_fold_summary_by_epoch.csv"), index=False)
        plot_training_curves(summary_df, os.path.join(Config.OUTPUT_DIR, "k_fold_summary_by_epoch.png"))

    print("\nAll done.")


if __name__ == "__main__":
    train()
