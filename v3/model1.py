import torch
import torch.nn as nn

class YamamotoPitchRNN(nn.Module):
    def __init__(
        self,
        num_pitch_types,
        num_prev_pitch_tokens,
        num_batter_hands,
        num_prev_result_tokens,
        input_numeric_dim,
        hidden_size=128,
        num_layers=2,
        dropout=0.20,
        prev_pitch_emb_dim=16,
        batter_hand_emb_dim=4,
        prev_result_emb_dim=8,
    ):
        super().__init__()

        # -------- Embeddings --------
        self.prev_pitch_emb  = nn.Embedding(num_prev_pitch_tokens, prev_pitch_emb_dim)
        self.batter_hand_emb = nn.Embedding(num_batter_hands, batter_hand_emb_dim)
        self.prev_result_emb = nn.Embedding(num_prev_result_tokens, prev_result_emb_dim)

        # -------- Pre-encoder MLP --------
        combined_in_dim = (
            prev_pitch_emb_dim
            + batter_hand_emb_dim
            + prev_result_emb_dim
            + input_numeric_dim
        )
        self.pre_mlp = nn.Sequential(
            nn.Linear(combined_in_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # -------- GRU backbone --------
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # -------- LayerNorm --------
        self.layernorm = nn.LayerNorm(hidden_size)

        # -------- Optional Attention --------
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=2,
            dropout=dropout,
            batch_first=True
        )

        # -------- Output head --------
        self.fc_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_pitch_types)
        )

    def forward(self, prev_pitch_idx, batter_hand_idx, prev_result_idx, numeric_feats):
        # embeddings
        prev_pitch_e  = self.prev_pitch_emb(prev_pitch_idx)
        batter_hand_e = self.batter_hand_emb(batter_hand_idx)
        prev_result_e = self.prev_result_emb(prev_result_idx)

        x = torch.cat([prev_pitch_e, batter_hand_e, prev_result_e, numeric_feats], dim=-1)

        # non-linear mixing before sequence model
        x = self.pre_mlp(x)

        # GRU
        rnn_out, _ = self.rnn(x)
        rnn_out = self.layernorm(rnn_out)

        # lightweight self-attention
        attn_out, _ = self.attn(rnn_out, rnn_out, rnn_out)

        # predict pitch for each time step
        logits = self.fc_out(attn_out)

        return logits
