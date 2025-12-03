# ============================
# train.py (Leak-Proof Version)
# ============================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import (
    DEVICE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    GRAD_CLIP,
    NUMERIC_COLS,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
)
from dataset import YamaPADataset, yama_collate
from model import YamamotoPitchRNN

# -------------------------------------------------
# Extra numeric features: batter stats vs pitch type
# -------------------------------------------------
BATTER_STAT_COLS = [
    "AVG_FF", "AVG_FS", "AVG_CU", "AVG_FC", "AVG_SI", "AVG_SL",
    "OBP_FF", "OBP_FS", "OBP_CU", "OBP_FC", "OBP_SI", "OBP_SL",
    "SLG_FF", "SLG_FS", "SLG_CU", "SLG_FC", "SLG_SI", "SLG_SL",
]

ALL_NUMERIC_COLS = NUMERIC_COLS + BATTER_STAT_COLS

# -------------------------------------------------
# Reproducibility + regularization hyperparams
# -------------------------------------------------
RANDOM_SEED   = 42
WEIGHT_DECAY  = 1e-4
LABEL_SMOOTH  = 0.1
PATIENCE      = 8

BEST_MODEL_PATH = "yamamoto_rnn_v3_best.pt"
LOSS_CURVE_PATH = "loss_curve_v3.png"
ACC_CURVE_PATH  = "accuracy_curve_v3.png"

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# -------------------------------------------------
# Load pitch data and merge batter stats
# -------------------------------------------------
df = pd.read_csv("yamamoto_v3_pitches_2025.csv")
batter_stats = pd.read_csv("batter_stats.csv")

expected_cols = {"batter_name", "batter_id"} | set(BATTER_STAT_COLS)
missing = expected_cols - set(batter_stats.columns)
if missing:
    raise ValueError(f"batter_stats.csv missing columns: {missing}")

df = df.merge(
    batter_stats,
    how="left",
    left_on="batter",
    right_on="batter_id",
)

df[BATTER_STAT_COLS] = df[BATTER_STAT_COLS].astype(float).fillna(df[BATTER_STAT_COLS].mean())
df[NUMERIC_COLS] = df[NUMERIC_COLS].astype(float)

# -------------------------------------------------
# Pitch Distribution
# -------------------------------------------------
pitch_counts = df["pitch_type"].value_counts(normalize=True).sort_index()
pitch_counts_raw = df["pitch_type"].value_counts().sort_index()

print("\n--- TRUE PITCH DISTRIBUTION (WITH BATTER STATS) ---")
for pitch in pitch_counts.index:
    pct = 100 * pitch_counts[pitch]
    raw = pitch_counts_raw[pitch]
    print(f"{pitch:>3s} : {pct:6.2f}% ({raw} pitches)")

# -------------------------------------------------
# Train / Validation split (LEAK-PROOF + SAVED)
# -------------------------------------------------
game_ids = df["game_pk"].unique()
np.random.shuffle(game_ids)

split_idx = int(0.8 * len(game_ids))
train_games = set(game_ids[:split_idx])
val_games   = set(game_ids[split_idx:])

# >>> Save splits so evaluate.py uses EXACT SAME SET <<<
np.save("train_games.npy", np.array(list(train_games)))
np.save("val_games.npy", np.array(list(val_games)))
print(f"\nSaved train_games.npy and val_games.npy ({len(train_games)} train games, {len(val_games)} val games).")

train_df = df[df["game_pk"].isin(train_games)].copy()
val_df   = df[df["game_pk"].isin(val_games)].copy()

train_pas = train_df[["game_pk", "at_bat_number"]].drop_duplicates().to_numpy().tolist()
val_pas   = val_df[["game_pk", "at_bat_number"]].drop_duplicates().to_numpy().tolist()

train_dataset = YamaPADataset(train_df, train_pas, ALL_NUMERIC_COLS)
val_dataset   = YamaPADataset(val_df,   val_pas,   ALL_NUMERIC_COLS)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=yama_collate,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=yama_collate,
)

# -------------------------------------------------
# Model setup
# -------------------------------------------------
num_pitch_types        = int(df["pitch_type_idx"].max() + 1)
num_prev_pitch_tokens  = int(df["prev_pitch_idx"].max() + 1)
num_batter_hands       = int(df["batter_hand_idx"].max() + 1)
num_prev_result_tokens = int(df["prev_pitch_result_idx"].max() + 1)

model = YamamotoPitchRNN(
    num_pitch_types,
    num_prev_pitch_tokens,
    num_batter_hands,
    num_prev_result_tokens,
    input_numeric_dim=len(ALL_NUMERIC_COLS),
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(DEVICE)

# -------------------------------------------------
# Loss / Optimizer
# -------------------------------------------------
pitch_freq = df["pitch_type_idx"].value_counts().sort_index()
inv_freq   = 1.0 / pitch_freq.values
alpha      = 0.25
weights    = (inv_freq ** alpha)
weights    = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=weights,
    ignore_index=-100,
    label_smoothing=LABEL_SMOOTH,
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS
)

# -------------------------------------------------
# Training Loop
# -------------------------------------------------
train_losses, val_losses = [], []
train_accs, val_accs = [], []

best_val_loss = float("inf")
epochs_no_improve = 0

for epoch in range(1, NUM_EPOCHS + 1):

    # -------- Training --------
    model.train()
    total_loss = 0.0
    running_correct = 0
    running_total   = 0

    for batch in train_loader:
        prev_b, hand_b, prev_res_b, num_b, labels_b, _ = batch

        prev_b  = prev_b.to(DEVICE)
        hand_b  = hand_b.to(DEVICE)
        prev_res_b = prev_res_b.to(DEVICE)
        num_b   = num_b.to(DEVICE)
        labels_b = labels_b.to(DEVICE)

        logits = model(prev_b, hand_b, prev_res_b, num_b)
        B, T, C = logits.shape

        loss = criterion(logits.view(B*T, C), labels_b.view(B*T))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = labels_b != -100
            running_correct += (preds[mask] == labels_b[mask]).sum().item()
            running_total   += mask.sum().item()

    avg_train_loss = total_loss / len(train_loader)
    avg_train_acc  = running_correct / running_total
    train_losses.append(avg_train_loss)
    train_accs.append(avg_train_acc)

    # -------- Validation --------
    model.eval()
    val_loss = 0.0
    correct = 0
    total   = 0

    with torch.no_grad():
        for batch in val_loader:
            prev_b, hand_b, prev_res_b, num_b, labels_b, _ = batch
            prev_b  = prev_b.to(DEVICE)
            hand_b  = hand_b.to(DEVICE)
            prev_res_b = prev_res_b.to(DEVICE)
            num_b   = num_b.to(DEVICE)
            labels_b = labels_b.to(DEVICE)

            logits = model(prev_b, hand_b, prev_res_b, num_b)
            B, T, C = logits.shape

            loss = criterion(logits.view(B*T, C), labels_b.view(B*T))
            val_loss += loss.item()

            preds = logits.argmax(dim=-1)
            mask = labels_b != -100
            correct += (preds[mask] == labels_b[mask]).sum().item()
            total   += mask.sum().item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc  = correct / total

    val_losses.append(avg_val_loss)
    val_accs.append(avg_val_acc)

    scheduler.step()

    improved = avg_val_loss < best_val_loss - 1e-4
    if improved:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        flag = "*"
    else:
        epochs_no_improve += 1
        flag = " "

    print(
        f"Epoch {epoch:02d} | Train Loss {avg_train_loss:.4f} | "
        f"Val Loss {avg_val_loss:.4f} | Train Acc {avg_train_acc:.4f} | "
        f"Val Acc {avg_val_acc:.4f} {flag}"
    )

    if epochs_no_improve >= PATIENCE:
        print(f"\nEarly stopping after {epoch} epochs.")
        break

print(f"\nBest validation loss: {best_val_loss:.4f}")
print(f"Best model saved to {BEST_MODEL_PATH}")

# -------- Save curves --------
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Loss Curve")
plt.savefig(LOSS_CURVE_PATH, dpi=200)
plt.close()

plt.figure()
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.legend()
plt.title("Accuracy Curve")
plt.savefig(ACC_CURVE_PATH, dpi=200)
plt.close()

print(f"Saved curves: {LOSS_CURVE_PATH}, {ACC_CURVE_PATH}")
