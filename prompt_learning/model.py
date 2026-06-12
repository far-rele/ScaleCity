import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class SeparatePromptEncoder(nn.Module):
    """Encode scale and task descriptions separately, then fuse them.

    BERT stays frozen; learnable projection and interaction layers sharpen the
    prompt vector's discriminability.

    Input:  scale_emb [B, 768], task_emb [B, 768] (from frozen BERT)
    Output: fused_prompt [B, prompt_dim]
    """
    def __init__(self, bert_dim=768, prompt_dim=768):
        super().__init__()
        self.prompt_dim = prompt_dim
        half_dim = prompt_dim // 2

        # Independent projections into scale / task semantic subspaces
        self.scale_proj = nn.Sequential(
            nn.Linear(bert_dim, half_dim),
            nn.GELU(),
            nn.LayerNorm(half_dim)
        )
        self.task_proj = nn.Sequential(
            nn.Linear(bert_dim, half_dim),
            nn.GELU(),
            nn.LayerNorm(half_dim)
        )

        # Interaction layer learning the scale-task relationship
        self.fusion = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim),
            nn.GELU(),
            nn.LayerNorm(prompt_dim),
            nn.Linear(prompt_dim, prompt_dim)
        )

    def forward(self, scale_emb, task_emb):
        s = self.scale_proj(scale_emb)        # [B, 384]
        t = self.task_proj(task_emb)          # [B, 384]
        combined = torch.cat([s, t], dim=-1)  # [B, 768]
        return self.fusion(combined)          # [B, 768]


class HyperNetwork(nn.Module):
    """Generate per-task fusion and prediction-head parameters from the prompt
    vector (batch-parallel). Uses a shared condition encoder and low-rank
    weight factorization.
    """
    def __init__(self, task_dim, in_dim, out_dim):
        """
        task_dim: PROMPT_EMB_DIM, prompt vector dimension
        in_dim:   RAW_INPUT_DIM, pretrained embedding dimension (512)
        out_dim:  HIDDEN_DIM, hidden dimension (256)
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_matrices = 3  # 3 modalities (SVI, POI, AE)
        self.rank = 16         # rank of the low-rank factorization
        self.cond_dim = 256    # condition feature dimension

        # Shared condition encoder: compress the 768-d prompt into a condition vector
        self.task_encoder = nn.Sequential(
            nn.Linear(task_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, self.cond_dim),
            nn.GELU(),
            nn.LayerNorm(self.cond_dim)
        )

        # Gate coefficients alpha (softmax over modalities)
        self.gate_net = nn.Sequential(
            nn.Linear(self.cond_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3)
        )

        # Low-rank weight generators: W = A @ B
        # Generator A: [B, cond] -> [B, 3, In, Rank]
        self.generator_A = nn.Sequential(
            nn.Linear(self.cond_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, self.num_matrices * in_dim * self.rank)
        )
        # Generator B: [B, cond] -> [B, 3, Rank, Out]
        self.generator_B = nn.Sequential(
            nn.Linear(self.cond_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, self.num_matrices * self.rank * out_dim)
        )
        # Bias generator: [B, 3, Out]
        self.generator_bias = nn.Sequential(
            nn.Linear(self.cond_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.num_matrices * out_dim)
        )

        # Prediction-head parameter generators
        self.pred_hidden_dim = 64
        # W_pred: [B, 64, 1]
        self.generator_pred_W = nn.Sequential(
            nn.Linear(self.cond_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, self.pred_hidden_dim * 1)
        )
        # b_pred: [B, 1, 1]
        self.generator_pred_b = nn.Sequential(
            nn.Linear(self.cond_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, task_emb):
        """
        Args:
            task_emb: [B, task_dim]
        Returns:
            gates, weights, biases, pred_W, pred_b
        """
        batch_size = task_emb.size(0)

        cond = self.task_encoder(task_emb)  # [B, cond_dim]

        # Gates: [B, 3, 1, 1] for broadcasting
        raw_gates = self.gate_net(cond)
        gates = F.softmax(raw_gates, dim=-1).view(batch_size, 3, 1, 1)

        # Weights: W = A @ B -> [B, 3, In, Out]
        params_A = self.generator_A(cond).view(batch_size, self.num_matrices, self.in_dim, self.rank)
        params_B = self.generator_B(cond).view(batch_size, self.num_matrices, self.rank, self.out_dim)
        weights = torch.matmul(params_A, params_B)

        # Biases: [B, 3, Out]
        biases = self.generator_bias(cond).view(batch_size, self.num_matrices, self.out_dim)

        # Prediction-head params
        pred_W = self.generator_pred_W(cond).view(batch_size, self.pred_hidden_dim, 1)
        pred_b = self.generator_pred_b(cond).view(batch_size, 1, 1)

        return gates, weights, biases, pred_W, pred_b


class UrbanPromptModel(nn.Module):
    """Prompt-conditioned multi-scale prediction network.

    Combines: multimodal fusion, dual-stream architecture, HyperNetwork-generated
    weights, area-weighted aggregation, and a dynamic prediction head. The prompt
    vector is produced by a separate scale/task encoder.
    """
    def __init__(self):
        super(UrbanPromptModel, self).__init__()

        self.raw_dim = Config.RAW_INPUT_DIM     # 512
        self.hidden_dim = Config.HIDDEN_DIM     # 256
        self.prompt_dim = Config.PROMPT_EMB_DIM # 768
        self.pred_hidden_dim = 64               # prediction-head hidden dim

        # Separate prompt encoder
        self.prompt_encoder = SeparatePromptEncoder(
            bert_dim=self.prompt_dim, prompt_dim=self.prompt_dim
        )

        # HyperNetwork (fusion layer dims: 512 -> 256)
        self.hypernet = HyperNetwork(self.prompt_dim, self.raw_dim, self.hidden_dim)

        # Static stream: an independent projection per modality
        self.static_residual_projs = nn.ModuleList([
            nn.Linear(self.raw_dim, self.hidden_dim),  # SVI
            nn.Linear(self.raw_dim, self.hidden_dim),  # POI
            nn.Linear(self.raw_dim, self.hidden_dim)   # AE
        ])
        # Static fusion: concat of 3 modalities [3 * 256] -> [256]
        self.static_fusion = nn.Linear(self.hidden_dim * 3, self.hidden_dim)

        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        # Prediction head input includes the concatenated total area (+1)
        self.pred_input_dim = self.hidden_dim + 1

        # Shared feature extractor
        self.shared_predictor_hidden = nn.Sequential(
            nn.Linear(self.pred_input_dim, self.pred_hidden_dim),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT)
        )

    def forward(self, feats, weights, masks, scale_emb, task_type_emb):
        """
        Args:
            feats:         [B, Max_Len, 3, 512] input features
            weights:       [B, Max_Len] physical overlap ratio (overlap/area)
            masks:         [B, Max_Len] padding mask
            scale_emb:     [B, 768] BERT encoding of the scale description
            task_type_emb: [B, 768] BERT encoding of the task description
        """
        batch_size, _, num_modal, _ = feats.shape

        # 1. Prompt encoding + HyperNetwork parameter generation
        fused_prompt = self.prompt_encoder(scale_emb, task_type_emb)
        # gates:[B,3,1,1]  W:[B,3,512,256]  b:[B,3,256]  pred_W:[B,64,1]  pred_b:[B,1,1]
        gates, W, b, pred_W, pred_b = self.hypernet(fused_prompt)

        # 2. Dynamic stream: feats [B, L, 3, 512] x W [B, 3, 512, 256]
        # einsum 'blmi,bmio->blmo': b=Batch, l=Len, m=Modality, i=In(512), o=Out(256)
        projected = torch.einsum('blmi,bmio->blmo', feats, W)
        projected = projected + b.unsqueeze(1)                  # broadcast bias [B,1,3,256]
        gates_expanded = gates.view(batch_size, 1, num_modal, 1)  # [B,1,3,1]
        weighted = projected * gates_expanded
        dynamic_feat = torch.sum(weighted, dim=2)               # [B, L, 256]

        # 3. Static stream
        static_feats_list = []
        for i in range(num_modal):
            feat_i = feats[:, :, i, :]                          # [B, L, 512]
            static_feats_list.append(self.static_residual_projs[i](feat_i))
        concat_feat = torch.cat(static_feats_list, dim=-1)      # [B, L, 768]
        static_feat = self.static_fusion(concat_feat)           # [B, L, 256]

        # 4. Dual-stream fusion
        fused_grid_feats = self.layer_norm(dynamic_feat + static_feat)

        # 5. Area-weighted region aggregation
        w_expanded = weights.unsqueeze(-1)  # [B, L, 1]
        m_expanded = masks.unsqueeze(-1)    # [B, L, 1]
        total_area = torch.sum(weights * masks, dim=1, keepdim=True)
        sum_emb = torch.sum(fused_grid_feats * w_expanded * m_expanded, dim=1)
        mean_emb = sum_emb / (total_area + 1e-6)
        input_vec = torch.cat([mean_emb, total_area], dim=1)

        # 6. Prediction head (dynamic params from HyperNetwork)
        hidden_feat = self.shared_predictor_hidden(input_vec)
        hidden_feat_expanded = hidden_feat.unsqueeze(1)         # [B, 1, 64]
        prediction = torch.bmm(hidden_feat_expanded, pred_W)    # [B, 1, 1]
        prediction = prediction + pred_b
        return prediction.squeeze(-1).squeeze(-1)               # [B]
