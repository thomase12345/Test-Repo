import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import DEVICE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, GRAD_CLIP, NUMERIC_COLS, HIDDEN_SIZE, NUM_LAYERS, DROPOUT
from dataset import YamaPADataset, yama_collate
from model import YamamotoPitchRNN

df = pd.read_csv("yamamoto_v3_pitches_2025.csv")
df[NUMERIC_COLS] = df[NUMERIC_COLS].astype(float)

# -----------------------------
# Pitch Type Distribution (Ground Truth)
# -----------------------------

pitch_counts = df["pitch_type"].value_counts(normalize=True).sort_index()
pitch_counts_raw = df["pitch_type"].value_counts().sort_index()

print("\n--- TRUE PITCH DISTRIBUTION (V2) ---")
for pitch in pitch_counts.index:
    pct = 100 * pitch_counts[pitch]
    raw = pitch_counts_raw[pitch]
    print(f"{pitch:>3s} : {pct:6.2f}%  ({raw} pitches)")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

game_ids = df["game_pk"].unique()
np.random.shuffle(game_ids)

split_idx = int(0.8 * len(game_ids))
train_games = set(game_ids[:split_idx])
val_games   = set(game_ids[split_idx:])

train_df = df[df["game_pk"].isin(train_games)].copy()
val_df   = df[df["game_pk"].isin(val_games)].copy()

train_pas = train_df[["game_pk", "at_bat_number"]].drop_duplicates().to_numpy().tolist()
val_pas   = val_df[["game_pk", "at_bat_number"]].drop_duplicates().to_numpy().tolist()

train_dataset = YamaPADataset(train_df, train_pas, NUMERIC_COLS)
val_dataset   = YamaPADataset(val_df, val_pas, NUMERIC_COLS)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=yama_collate)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=yama_collate)

num_pitch_types = int(df["pitch_type_idx"].max() + 1)
num_prev_pitch_tokens = int(df["prev_pitch_idx"].max() + 1)
num_batter_hands = int(df["batter_hand_idx"].max() + 1)
num_prev_result_tokens = int(df["prev_pitch_result_idx"].max() + 1)

model = YamamotoPitchRNN(
    num_pitch_types,
    num_prev_pitch_tokens,
    num_batter_hands,
    num_prev_result_tokens,
    input_numeric_dim=len(NUMERIC_COLS),
    hidden_size = HIDDEN_SIZE,
    num_layers = NUM_LAYERS,
    dropout = DROPOUT,
).to(DEVICE)

pitch_freq = df["pitch_type_idx"].value_counts().sort_index()
inv_freq = 1.0 / pitch_freq.values
alpha = 0.5 # was 0.5
weights = inv_freq ** alpha  # sqrt of inverse frequency
weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)


criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
val_losses = []

for epoch in range(1, NUM_EPOCHS + 1):
    # -----------------------------
    # Train phase
    # -----------------------------
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        prev_b, hand_b, prev_res_b, num_b, labels_b, _ = batch
        prev_b, hand_b, prev_res_b, num_b, labels_b = (
            prev_b.to(DEVICE), hand_b.to(DEVICE), prev_res_b.to(DEVICE),
            num_b.to(DEVICE), labels_b.to(DEVICE)
        )

        logits = model(prev_b, hand_b, prev_res_b, num_b)
        B, T, C = logits.shape

        loss = criterion(
            logits.view(B * T, C),
            labels_b.view(B * T)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # -----------------------------
    # Validation: loss + accuracy
    # -----------------------------
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            prev_b, hand_b, prev_res_b, num_b, labels_b, _ = batch
            prev_b, hand_b, prev_res_b, num_b, labels_b = (
                prev_b.to(DEVICE), hand_b.to(DEVICE), prev_res_b.to(DEVICE),
                num_b.to(DEVICE), labels_b.to(DEVICE)
            )

            logits = model(prev_b, hand_b, prev_res_b, num_b)
            B, T, C = logits.shape

            loss = criterion(
                logits.view(B * T, C),
                labels_b.view(B * T)
            )
            val_loss += loss.item()

            preds = logits.argmax(dim=-1)
            valid_mask = labels_b != -100
            correct += (preds[valid_mask] == labels_b[valid_mask]).sum().item()
            total += valid_mask.sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    val_acc = correct / total if total > 0 else 0.0

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

torch.save(model.state_dict(), "yamamoto_rnn_v3.pt")


plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.savefig("loss_curve_v3.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved loss curve to loss_curve_v2.png")