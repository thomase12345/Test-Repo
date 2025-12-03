import torch

# -----------------------------
# Device
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Training params
# -----------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 22
LEARNING_RATE = 1e-3 #prev: 5e-4 (slow)
GRAD_CLIP = 1.0

# -----------------------------
# Model params
# -----------------------------
HIDDEN_SIZE = 64 #default is 64
NUM_LAYERS = 2 # default is 2 (we are doing 2 layer GRU)
DROPOUT = 0 #default is 0.25, note if you set to 1-layer GRU, model disregards dropout

# -----------------------------
# Numeric feature columns (V2)
# -----------------------------
NUMERIC_COLS = [
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "is_top_inning",
    "on_1b_flag",
    "on_2b_flag",
    "on_3b_flag",
    "score_diff_pov",
    "pitcher_ahead_flag",
    "hitter_ahead_flag",
    "putaway_count_flag",
    "platoon_adv",
    "fastballs_in_pa",
    "last_two_fastballs_flag",
    "risp_flag",
    "high_leverage_flag",
    "prev_release_speed",
    "prev_pfx_x",
    "prev_pfx_z",
    "prev_speed_minus_ff_mean",
]