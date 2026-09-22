# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, AutoModel
from config import Config
from peft import LoraConfig, get_peft_model


class GraphAttentionLayer(nn.Module):
    """Single graph attention (GAT) layer.

    Input:  features [N, In_Dim], adjacency [N, N]
    Output: features [N, Out_Dim]
    """
    def __init__(self, in_features, out_features, dropout, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        # h: [N, in_features], adj: [N, N] (0 or 1)
        N = h.size(0)
        wh = self.W(h)  # [N, out_features]

        # Build all node pairs via broadcasting, then compute attention coefficients
        a_input = torch.cat([wh.repeat(1, N).view(N * N, -1),
                             wh.repeat(N, 1)], dim=1).view(N, N, 2 * self.out_features)

        e = self.leakyrelu(self.a(a_input).squeeze(2))  # [N, N]

        # Mask: set e to -inf where adj == 0
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        # Aggregate neighbor information
        h_prime = torch.matmul(attention, wh)  # [N, out]

        return F.elu(h_prime)


class GATBlock(nn.Module):
    """Multi-head GAT block with residual connection."""
    def __init__(self, in_dim, hidden_dim, num_heads, dropout):
        super(GATBlock, self).__init__()
        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_dim, hidden_dim, dropout=dropout)
            for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(hidden_dim * num_heads, in_dim)  # project back for residual
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, h, adj):
        head_outs = [head(h, adj) for head in self.heads]
        h_cat = torch.cat(head_outs, dim=1)  # [N, hidden * heads]

        h_res = self.out_proj(h_cat)
        h_res = self.dropout(h_res)
        return self.norm(h + h_res)


class UrbanGraphModel(nn.Module):
    def __init__(self):
        super(UrbanGraphModel, self).__init__()

        # 1. Image encoder (CLIP) + LoRA + gradient checkpointing
        print(f"Loading CLIP: {Config.CLIP_MODEL_NAME}...")
        self.clip_vision = CLIPVisionModel.from_pretrained(Config.CLIP_MODEL_NAME)
        self.img_hidden_dim = self.clip_vision.config.hidden_size

        # Gradient checkpointing trades compute for memory during backprop
        self.clip_vision.enable_input_require_grads()
        self.clip_vision.gradient_checkpointing_enable()

        # Inject LoRA into CLIP projection / FC layers
        clip_lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            lora_dropout=0.1,
            bias="none"
        )
        self.clip_vision = get_peft_model(self.clip_vision, clip_lora_config)
        self.clip_vision.print_trainable_parameters()

        # 2. Text encoder + LoRA + gradient checkpointing
        print(f"Loading Text Encoder: {Config.text_MODEL_NAME}...")
        self.text_model = AutoModel.from_pretrained(Config.text_MODEL_NAME)
        self.txt_hidden_dim = self.text_model.config.hidden_size

        self.text_model.enable_input_require_grads()
        self.text_model.gradient_checkpointing_enable()

        # Inject LoRA into the attention and dense layers
        text_lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "key", "value", "dense"],
            lora_dropout=0.1,
            bias="none"
        )
        self.text_model = get_peft_model(self.text_model, text_lora_config)
        # Freeze the unused pooler (and its LoRA params)
        for name, param in self.text_model.named_parameters():
            if "pooler" in name:
                param.requires_grad = False
        self.text_model.print_trainable_parameters()

        # 3. GAT layers (independent graph networks for SVI and POI)
        self.svi_gats = nn.ModuleList([
            GATBlock(self.img_hidden_dim, self.img_hidden_dim // Config.GAT_HEADS, Config.GAT_HEADS, Config.GAT_DROPOUT)
            for _ in range(Config.GAT_LAYERS)
        ])

        self.poi_gats = nn.ModuleList([
            GATBlock(self.txt_hidden_dim, self.txt_hidden_dim // Config.GAT_HEADS, Config.GAT_HEADS, Config.GAT_DROPOUT)
            for _ in range(Config.GAT_LAYERS)
        ])

        # AE branch: Input(64) -> Lifting MLP(256) -> GAT -> Proj -> Embed(512)
        self.ae_lifting = nn.Sequential(
            nn.Linear(Config.AE_INPUT_DIM, Config.AE_LIFTING_DIM),
            nn.LayerNorm(Config.AE_LIFTING_DIM),
            nn.GELU(),
            nn.Dropout(Config.GAT_DROPOUT)
        )
        self.ae_gats = nn.ModuleList([
            GATBlock(
                in_dim=Config.AE_LIFTING_DIM,
                hidden_dim=Config.AE_LIFTING_DIM // Config.GAT_HEADS,
                num_heads=Config.GAT_HEADS,
                dropout=Config.GAT_DROPOUT
            )
            for _ in range(Config.GAT_LAYERS)
        ])

        # 4. Projection heads (map each modality into the common space)
        self.svi_proj = nn.Sequential(
            nn.Linear(self.img_hidden_dim, self.img_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.img_hidden_dim, Config.EMBED_DIM)
        )

        self.poi_proj = nn.Sequential(
            nn.Linear(self.txt_hidden_dim, self.txt_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.txt_hidden_dim, Config.EMBED_DIM)
        )

        self.ae_proj = nn.Sequential(
            nn.Linear(Config.AE_LIFTING_DIM, Config.AE_LIFTING_DIM),
            nn.ReLU(),
            nn.Linear(Config.AE_LIFTING_DIM, Config.EMBED_DIM)
        )

    def encode_images(self, images, mask):
        # images: [N, M, C, H, W], mask: [N, M] (0 or 1)
        N, M, C, H, W = images.shape

        # Run CLIP only on valid (mask=1) images, skipping zero-padded ones
        flat_imgs = images.view(-1, C, H, W)    # [N*M, C, H, W]
        flat_mask = mask.view(-1).bool()         # [N*M]

        valid_imgs = flat_imgs[flat_mask]
        outputs = self.clip_vision(pixel_values=valid_imgs)
        valid_feats = outputs.pooler_output      # [K, 768]

        # Scatter valid features back, leaving padded positions at 0
        feat_dim = valid_feats.shape[-1]
        img_feats_flat = torch.zeros(N * M, feat_dim, device=images.device, dtype=valid_feats.dtype)
        img_feats_flat[flat_mask] = valid_feats
        img_feats = img_feats_flat.view(N, M, -1)  # [N, M, 768]

        # Masked mean pooling
        mask_expanded = mask.unsqueeze(-1).to(img_feats.device)
        masked_feats = img_feats * mask_expanded
        sum_feats = masked_feats.sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1e-9)
        grid_feats = sum_feats / count
        return grid_feats

    def encode_texts(self, input_ids, attention_mask, grid_mask):
        # input_ids: [N, M, L], grid_mask: [N, M] (for grid aggregation)
        N, M, L = input_ids.shape
        flat_ids = input_ids.view(-1, L)
        flat_mask = attention_mask.view(-1, L)

        outputs = self.text_model(input_ids=flat_ids, attention_mask=flat_mask)
        txt_feats = outputs.last_hidden_state[:, 0, :]  # [N*M, 768]
        txt_feats = txt_feats.view(N, M, -1)

        # Masked mean pooling
        mask_expanded = grid_mask.unsqueeze(-1).to(txt_feats.device)
        masked_feats = txt_feats * mask_expanded
        sum_feats = masked_feats.sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1e-9)

        grid_feats = sum_feats / count
        return grid_feats

    def forward(self, batch_data):
        # 1. Base encoding (data assumed already on device)
        svi_base = self.encode_images(batch_data['svi'], batch_data['svi_mask'])
        poi_base = self.encode_texts(batch_data['poi_ids'], batch_data['poi_mask'], batch_data['poi_grid_mask'])
        ae_base = batch_data['ae']

        adj = batch_data['adj']  # [N, N]

        # 2. Graph enhancement via GAT
        svi_graph = svi_base
        for layer in self.svi_gats:
            svi_graph = layer(svi_graph, adj)

        poi_graph = poi_base
        for layer in self.poi_gats:
            poi_graph = layer(poi_graph, adj)

        # AE: lift [N, 64] -> [N, 256] then GAT
        ae_lifted = self.ae_lifting(ae_base)
        ae_graph = ae_lifted
        for layer in self.ae_gats:
            ae_graph = layer(ae_graph, adj)

        # 3. Project to the common space
        z_svi = self.svi_proj(svi_graph)
        z_poi = self.poi_proj(poi_graph)
        z_ae = self.ae_proj(ae_graph)

        # 4. L2 normalize (important for contrastive learning)
        z_svi = F.normalize(z_svi, p=2, dim=1)
        z_poi = F.normalize(z_poi, p=2, dim=1)
        z_ae = F.normalize(z_ae, p=2, dim=1)

        return z_svi, z_poi, z_ae
