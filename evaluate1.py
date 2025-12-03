# ==============================
# evaluate.py (Leak-Proof Version)
# ==============================

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from config import DEVICE, NUMERIC_COLS, HIDDEN_SIZE, NUM_LAYERS, DROPOUT
from dataset import YamaPADataset, yama_collate
from model import YamamotoPitchRNN

BATTER_STAT_COLS = [
    "AVG_FF", "AVG_FS", "AVG_CU", "AVG_FC", "AVG_SI", "AVG_SL",
    "OBP_FF", "OBP_FS", "OBP_CU", "OBP_FC", "OBP_SI", "OBP_SL",
    "SLG_FF", "SLG_FS", "SLG_CU", "SLG_FC", "SLG_SI", "SLG_SL",
]

ALL_NUMERIC_COLS = NUMERIC_COLS + BATTER_STAT_COLS

# -----------------------------
# Load pitch data + batter stats
# -----------------------------
df = pd.read_csv("yamamoto_v3_pitches_2025.csv")
batter_stats = pd.read_csv("batter_stats.csv")

df = df.merge(
    batter_stats,
    how="left",
    left_on="batter",
    right_on="batter_id",
)

df[BATTER_STAT_COLS] = df[BATTER_STAT_COLS].astype(float).fillna(df[BATTER_STAT_COLS].mean())
df[NUMERIC_COLS]     = df[NUMERIC_COLS].astype(float)

# -----------------------------
# Load EXACT train/val split
# -----------------------------
train_games = set(np.load("train_games.npy"))
val_games   = set(np.load("val_games.npy"))
print(f"Loaded split: {len(train_games)} train games, {len(val_games)} val games.")

val_df = df[df["game_pk"].isin(val_games)].copy()
val_pas = val_df[["game_pk", "at_bat_number"]].drop_duplicates().to_numpy().tolist()

val_dataset = YamaPADataset(val_df, val_pas, ALL_NUMERIC_COLS)
val_loader  = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=yama_collate,
)

# -----------------------------
# Model setup & load best model
# -----------------------------
num_pitch_types       = int(df["pitch_type_idx"].max() + 1)
num_prev_pitch_tokens = int(df["prev_pitch_idx"].max() + 1)
num_batter_hands      = int(df["batter_hand_idx"].max() + 1)
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

model.load_state_dict(torch.load("yamamoto_rnn_v3_best.pt", map_location=DEVICE))
model.eval()

# -----------------------------
# Inference
# -----------------------------
all_preds  = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        prev_b, hand_b, prev_res_b, num_b, labels_b, _ = batch

        prev_b  = prev_b.to(DEVICE)
        hand_b  = hand_b.to(DEVICE)
        prev_res_b = prev_res_b.to(DEVICE)
        num_b   = num_b.to(DEVICE)

        logits = model(prev_b, hand_b, prev_res_b, num_b)
        preds  = logits.argmax(dim=-1).cpu()
        labels = labels_b.cpu()

        valid_mask = labels != -100
        all_preds.extend(preds[valid_mask].numpy())
        all_labels.extend(labels[valid_mask].numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# -----------------------------
# Accuracy + per-pitch breakdown
# -----------------------------
unique_classes = sorted(df["pitch_type_idx"].unique())
class_names    = [df[df["pitch_type_idx"] == i]["pitch_type"].iloc[0] for i in unique_classes]

overall_acc = (all_preds == all_labels).mean()
print(f"\nOVERALL VALIDATION ACCURACY: {overall_acc:.4f}")

print("\nPER-PITCH ACCURACY:")
for i, name in zip(unique_classes, class_names):
    mask = all_labels == i
    if mask.sum() == 0:
        acc = float("nan")
    else:
        acc = (all_preds[mask] == all_labels[mask]).mean()
    print(f"{name:>3s} : {acc:.4f}  (n={mask.sum()})")

# -----------------------------
# Confusion matrix
# -----------------------------
cm = confusion_matrix(all_labels, all_preds, labels=unique_classes)
fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, xticks_rotation="vertical")
plt.title("Confusion Matrix — Pitch Type Prediction")
plt.savefig("confusion_matrix_v3.png", dpi=200, bbox_inches="tight")
plt.close()

print("Saved confusion matrix to confusion_matrix_v3.png")
