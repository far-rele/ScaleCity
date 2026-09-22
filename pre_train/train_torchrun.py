# train_torchrun.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from tqdm import tqdm
import os
import shutil
import numpy as np
from transformers import CLIPProcessor, AutoTokenizer
from datetime import datetime
import math
import pickle

from config import Config
from data_utils import load_and_clean_data, UrbanDataset, UrbanGraphCollator
from models import UrbanGraphModel

import pandas as pd
import matplotlib.pyplot as plt

# Non-interactive backend for headless servers
plt.switch_backend('Agg')


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all processes with gradient support across processes."""
    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def gather_from_all_gpus(tensor):
    """Gather and concatenate a tensor from all GPUs (gradients preserved)."""
    if not dist.is_initialized():
        return tensor

    gathered_tensors = GatherLayer.apply(tensor)
    gathered_tensor = torch.cat(gathered_tensors, dim=0)
    return gathered_tensor


def save_and_plot_loss(loss_history, output_dir="./"):
    """Save the per-epoch loss to CSV and plot the curve."""
    df = pd.DataFrame({
        'epoch': range(1, len(loss_history) + 1),
        'loss': loss_history
    })
    csv_path = os.path.join(output_dir, "training_loss.csv")
    df.to_csv(csv_path, index=False)
    print(f"Loss data saved to: {csv_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['loss'], marker='o', linestyle='-', color='b', label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    img_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(img_path)
    print(f"Loss curve saved to: {img_path}")
    plt.close()


def setup_distributed():
    """Initialize the distributed environment (env vars set by torchrun)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return local_rank, rank, world_size
    else:
        print("No distributed environment detected, using single-GPU mode.")
        return 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def global_contrastive_loss(local_z1, local_z2, temperature=0.07):
    """Contrastive loss between local features and globally gathered features.

    local_z1, local_z2: current GPU features [B, D]
    """
    global_z1 = gather_from_all_gpus(local_z1)
    global_z2 = gather_from_all_gpus(local_z2)

    # Similarity of current-GPU samples vs all global samples [B, B * world_size]
    logits_1 = torch.matmul(local_z1, global_z2.T) / temperature
    logits_2 = torch.matmul(local_z2, global_z1.T) / temperature

    # Positive-sample index for sample i on this rank is rank * B + i
    B = local_z1.size(0)
    rank = dist.get_rank()
    labels = torch.arange(B, dtype=torch.long, device=local_z1.device) + rank * B

    loss_1 = nn.CrossEntropyLoss()(logits_1, labels)
    loss_2 = nn.CrossEntropyLoss()(logits_2, labels)

    return (loss_1 + loss_2) / 2


def contrastive_loss(z1, z2, temperature=0.07):
    """Single-GPU contrastive loss."""
    sim_matrix = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    loss_1 = nn.CrossEntropyLoss()(sim_matrix, labels)
    loss_2 = nn.CrossEntropyLoss()(sim_matrix.T, labels)
    return (loss_1 + loss_2) / 2


def reduce_mean(tensor, nprocs):
    """Average a scalar across all GPUs."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def main():
    # 1. DDP init
    local_rank, global_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Per-process seed offset for reproducibility
    torch.manual_seed(Config.SEED + global_rank)
    np.random.seed(Config.SEED + global_rank)
    torch.backends.cudnn.benchmark = True

    # Output directory with timestamp
    output_dir = Config.PRE_TRAIN_OUTPUT
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, timestamp)
    if global_rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output dir: {output_dir}")
            config_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
            config_dst = os.path.join(output_dir, "config.py")
            shutil.copy2(config_src, config_dst)
            print(f"Backed up config: {config_dst}")
        print(f"Run timestamp: {timestamp}")
        print(f"=== Starting distributed training | GPUs: {world_size} ===")

    if global_rank == 0:
        print("=== Initializing data module ===")

    # Load data (every process loads it; fine on high-memory servers)
    valid_h3, svi_dict, poi_dict, ae_data, adj_dict = load_and_clean_data()

    if global_rank == 0:
        print("Loading processor and tokenizer...")

    processor = CLIPProcessor.from_pretrained(Config.CLIP_MODEL_NAME, local_files_only=False)
    tokenizer = AutoTokenizer.from_pretrained(Config.text_MODEL_NAME, local_files_only=False)

    dataset = UrbanDataset(valid_h3)
    collator = UrbanGraphCollator(svi_dict, poi_dict, ae_data, adj_dict, processor, tokenizer)

    # DistributedSampler splits data across GPUs
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,  # per-GPU batch size
        shuffle=False,                 # must be False when using a sampler
        num_workers=Config.NUM_WORKERS,
        collate_fn=collator,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=True
    )

    # 2. Model init
    model = UrbanGraphModel().to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Optimizer (separate LR for encoders vs heads); access raw model via .module
    raw_model = model.module
    encoder_params = list(map(id, raw_model.clip_vision.parameters())) + \
                     list(map(id, raw_model.text_model.parameters()))
    head_params = filter(lambda p: id(p) not in encoder_params, raw_model.parameters())

    optimizer = torch.optim.AdamW([
        {'params': raw_model.clip_vision.parameters(), 'lr': Config.LR_ENCODER},
        {'params': raw_model.text_model.parameters(), 'lr': Config.LR_ENCODER},
        {'params': head_params, 'lr': Config.LR_HEAD}
    ], weight_decay=Config.WEIGHT_DECAY)

    scaler = GradScaler()
    accumulation_steps = 4  # gradient accumulation to simulate a larger batch size

    loss_history = []
    best_loss = float('inf')
    best_model_path = None

    # 3. Training loop
    if global_rank == 0:
        print("=== Training ===")

    for epoch in range(Config.EPOCHS):
        # Re-seed the sampler each epoch for a different shuffle
        sampler.set_epoch(epoch)

        total_loss_epoch = 0
        optimizer.zero_grad()

        if global_rank == 0:
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        else:
            progress_bar = dataloader

        model.train()

        for i, batch in enumerate(progress_bar):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            with autocast():
                z_svi_all, z_poi_all, z_ae_all = model(batch)

                target_idx = batch['target_idx']
                z_svi = z_svi_all[target_idx]
                z_poi = z_poi_all[target_idx]
                z_ae = z_ae_all[target_idx]

                # Global (cross-GPU) contrastive loss across the three modality pairs
                loss_svi_poi = global_contrastive_loss(z_svi, z_poi)
                loss_svi_ae = global_contrastive_loss(z_svi, z_ae)
                loss_poi_ae = global_contrastive_loss(z_poi, z_ae)

                loss = (Config.LAMBDA_SVI_POI * loss_svi_poi +
                        Config.LAMBDA_SVI_AE * loss_svi_ae +
                        Config.LAMBDA_POI_AE * loss_poi_ae)

                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Reduce loss across GPUs for logging
            reduced_loss = loss.detach() * accumulation_steps
            global_avg_loss = reduce_mean(reduced_loss, world_size).item()
            total_loss_epoch += global_avg_loss
            if global_rank == 0:
                progress_bar.set_postfix({"Loss": f"{global_avg_loss:.4f}"})

        # Handle the trailing batches that don't fill an accumulation cycle
        if (i + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if global_rank == 0:
            avg_loss = total_loss_epoch / len(dataloader)
            print(f"Epoch {epoch+1} done. Avg Loss (Global): {avg_loss:.4f}")

            loss_history.append(avg_loss)
            print("Saving loss record and curve...")
            save_and_plot_loss(loss_history, output_dir=output_dir)

            if avg_loss < best_loss:
                print(f"Loss improved ({best_loss:.4f} -> {avg_loss:.4f}), saving model...")
                best_loss = avg_loss

                if best_model_path is not None and os.path.exists(best_model_path):
                    try:
                        os.remove(best_model_path)
                    except OSError as e:
                        print(f"Failed to remove old model: {e}")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = f"urban_best_{timestamp}_loss_{avg_loss:.4f}.pth"
                best_model_path = os.path.join(output_dir, model_filename)

                torch.save(model.module.state_dict(), best_model_path)
                print(f"Best model saved to: {best_model_path}")
            else:
                print(f"Loss did not improve (Best: {best_loss:.4f}), skipping save.")

        torch.cuda.empty_cache()

    # ==========================================================
    # After training: distributed feature extraction
    # ==========================================================
    if dist.is_initialized():
        dist.barrier()

    # Broadcast best_model_path so all ranks know where the model is
    path_list = [best_model_path] if global_rank == 0 else [None]
    if dist.is_initialized():
        dist.broadcast_object_list(path_list, src=0)
    best_model_path = path_list[0]

    if best_model_path and os.path.exists(best_model_path):
        if global_rank == 0:
            print("\n" + "=" * 50)
            print("=== Training done, starting distributed feature extraction ===")
            print(f"Loading best model on all ranks: {best_model_path}")
            print("=" * 50)

        # 1. Every rank loads the best weights
        raw_model.load_state_dict(torch.load(best_model_path, map_location=device))
        raw_model.eval()

        # 2. Distributed inference DataLoader (shuffle=False for reproducible order)
        inference_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
        inference_dataloader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collator,
            sampler=inference_sampler,
            pin_memory=True,
            persistent_workers=True
        )

        # 3. Reconstruct which H3 IDs this rank handles (replicate DistributedSampler padding)
        total_size = math.ceil(len(dataset) / world_size) * world_size
        indices = list(range(len(dataset)))
        indices += indices[:(total_size - len(indices))]
        local_indices = indices[global_rank:total_size:world_size]
        local_valid_h3 = [valid_h3[i] for i in local_indices]

        local_results = {}
        current_idx = 0

        if global_rank == 0:
            pbar = tqdm(inference_dataloader, desc="Extracting features")
        else:
            pbar = inference_dataloader

        # A modality is active if it participates in any contrastive loss (lambda > 0);
        # inactive modalities are saved as zeros.
        active_svi = (Config.LAMBDA_SVI_POI > 0) or (Config.LAMBDA_SVI_AE > 0)
        active_poi = (Config.LAMBDA_SVI_POI > 0) or (Config.LAMBDA_POI_AE > 0)
        active_ae = (Config.LAMBDA_SVI_AE > 0) or (Config.LAMBDA_POI_AE > 0)

        # 4. Parallel inference on all GPUs
        with torch.no_grad():
            for batch in pbar:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)

                with autocast():
                    z_svi_all, z_poi_all, z_ae_all = raw_model(batch)

                target_idx = batch['target_idx']
                z_svi = z_svi_all[target_idx].float().cpu().numpy()
                z_poi = z_poi_all[target_idx].float().cpu().numpy()
                z_ae = z_ae_all[target_idx].float().cpu().numpy()

                batch_len = len(z_svi)
                batch_h3_ids = local_valid_h3[current_idx: current_idx + batch_len]
                current_idx += batch_len

                for i, hid in enumerate(batch_h3_ids):
                    local_results[hid] = {
                        'svi_emb': z_svi[i] if active_svi else np.zeros_like(z_svi[i]),
                        'poi_emb': z_poi[i] if active_poi else np.zeros_like(z_poi[i]),
                        'ae_emb': z_ae[i] if active_ae else np.zeros_like(z_ae[i])
                    }

                torch.cuda.empty_cache()

        # 5. Gather all ranks' feature dicts and save on rank 0
        if global_rank == 0:
            print("Inference done on all ranks, gathering and merging features...")

        gathered_results = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_results, local_results)

        if global_rank == 0:
            final_results = {}
            for res_dict in gathered_results:
                final_results.update(res_dict)

            output_pkl = os.path.join(output_dir, f"{timestamp}.pkl")

            with open(output_pkl, 'wb') as f:
                pickle.dump(final_results, f)

            print("\nFeature extraction complete.")
            print(f"Merged {len(final_results)} grid embeddings, saved to:\n => {output_pkl}")
            print("=" * 50)

    else:
        if global_rank == 0:
            print("Best model not found, skipping feature extraction.")

    # Final barrier and cleanup
    if dist.is_initialized():
        dist.barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
