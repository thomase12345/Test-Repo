import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from config import DEVICE, NUMERIC_COLS, HIDDEN_SIZE, NUM_LAYERS, DROPOUT
from dataset import YamaPADataset, yama_collate
from model import YamamotoPitchRNN

df = pd.read_csv("yamamoto_v3_pitches_2025.csv")
df[NUMERIC_COLS] = df[NUMERIC_COLS].astype(float)

val_pas = df[["game_pk", "at_bat_number"]].drop_duplicates().to_numpy().tolist()
val_dataset = YamaPADataset(df, val_pas, NUMERIC_COLS)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=yama_collate)

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

model.load_state_dict(torch.load("yamamoto_rnn_v3.pt", map_location=DEVICE))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        prev_b, hand_b, prev_res_b, num_b, labels_b, _ = batch
        prev_b, hand_b, prev_res_b, num_b = (
            prev_b.to(DEVICE), hand_b.to(DEVICE), prev_res_b.to(DEVICE), num_b.to(DEVICE)
        )

        logits = model(prev_b, hand_b, prev_res_b, num_b)
        preds = logits.argmax(dim=-1).cpu()
        labels = labels_b.cpu()

        valid_mask = labels != -100
        all_preds.extend(preds[valid_mask].numpy())
        all_labels.extend(labels[valid_mask].numpy())

unique_classes = sorted(df["pitch_type_idx"].unique())
class_names = [df[df["pitch_type_idx"] == i]["pitch_type"].iloc[0] for i in unique_classes]

# -----------------------------
# Overall Accuracy
# -----------------------------

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

overall_acc = (all_preds == all_labels).mean()
print(f"\nOVERALL VALIDATION ACCURACY: {overall_acc:.4f}")

# -----------------------------
# Per-Class Accuracy
# -----------------------------

print("\nPER-PITCH ACCURACY:")
for i, pitch_name in zip(unique_classes, class_names):
    mask = all_labels == i
    if mask.sum() == 0:
        acc = float("nan")
    else:
        acc = (all_preds[mask] == all_labels[mask]).mean()
    print(f"{pitch_name:>3s} : {acc:.4f}  (n={mask.sum()})")

# -----------------------------
# Confusion Matrix (Saved)
# -----------------------------

cm = confusion_matrix(all_labels, all_preds, labels=unique_classes)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, xticks_rotation="vertical")
plt.title("Confusion Matrix: Pitch Type Prediction")

plt.savefig("confusion_matrix_v3.png", dpi=200, bbox_inches="tight")
plt.close()
print("\nSaved confusion matrix to confusion_matrix_v2.png")
