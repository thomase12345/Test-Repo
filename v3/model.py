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
        hidden_size=64,
        num_layers=2,
        dropout=0.25,
        prev_pitch_emb_dim=8,
        batter_hand_emb_dim=2,
        prev_result_emb_dim=4,
    ):
        super().__init__()

        self.prev_pitch_emb = nn.Embedding(num_prev_pitch_tokens, prev_pitch_emb_dim)
        self.batter_hand_emb = nn.Embedding(num_batter_hands, batter_hand_emb_dim)
        self.prev_result_emb = nn.Embedding(num_prev_result_tokens, prev_result_emb_dim)

        self.input_dim = (
            prev_pitch_emb_dim
            + batter_hand_emb_dim
            + prev_result_emb_dim
            + input_numeric_dim
        )

        self.rnn = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.fc_out = nn.Linear(hidden_size, num_pitch_types)

    def forward(self, prev_pitch_idx, batter_hand_idx, prev_result_idx, numeric_feats):
        prev_pitch_e = self.prev_pitch_emb(prev_pitch_idx)
        batter_hand_e = self.batter_hand_emb(batter_hand_idx)
        prev_result_e = self.prev_result_emb(prev_result_idx)

        x = torch.cat(
            [prev_pitch_e, batter_hand_e, prev_result_e, numeric_feats],
            dim=-1
        )

        rnn_out, _ = self.rnn(x)
        logits = self.fc_out(rnn_out)

        return logits