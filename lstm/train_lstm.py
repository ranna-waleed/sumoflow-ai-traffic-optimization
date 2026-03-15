
# lstm/train_lstm.py (BiLSTM)
import os, csv, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle, json
from collections import defaultdict

LSTM_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(LSTM_DIR, "data", "sequences.csv")
MODELS_DIR = os.path.join(LSTM_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

SEQ_LEN      = 60
PRED_LEN     = 30
FEATURES     = ["north", "south", "east", "west", "total", "avg_speed"]
TARGETS      = ["north", "south", "east", "west"]
HIDDEN       = 128
LAYERS       = 2
DROPOUT      = 0.2
EPOCHS       = 100
BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 20


class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class TrafficLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 output_size, pred_len, dropout, bidirectional=True):
        super().__init__()
        self.pred_len      = pred_len
        self.output_size   = output_size
        self.bidirectional = bidirectional
        self.directions    = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0,
            bidirectional = bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        # BiLSTM output is hidden_size * 2
        self.fc = nn.Linear(hidden_size * self.directions, output_size * pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1, :])   # last timestep (both directions)
        out    = self.fc(out)
        return out.view(-1, self.pred_len, self.output_size)


def load_data():
    print("[train] Loading data...")
    rows = []
    with open(DATA_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(row[k]) if k != "profile" else row[k] for k in row})

    by_profile = defaultdict(list)
    for r in rows:
        by_profile[r["profile"]].append(r)

    all_X, all_y = [], []
    for profile, data in by_profile.items():
        data.sort(key=lambda x: x["timestep"])
        arr = np.array([[r.get(f, 0.0) for f in FEATURES] for r in data], dtype=np.float32)
        for i in range(len(arr) - SEQ_LEN - PRED_LEN + 1):
            all_X.append(arr[i : i + SEQ_LEN])
            all_y.append(arr[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN, :4])

    X = np.array(all_X)
    y = np.array(all_y)
    print(f"[train] Sequences: {len(X)}, X={X.shape}, y={y.shape}")
    return X, y


def train():
    X, y = load_data()

    X_flat  = X.reshape(-1, len(FEATURES))
    min_val = X_flat.min(axis=0)
    max_val = X_flat.max(axis=0)
    denom   = max_val - min_val
    denom[denom == 0] = 1
    X_flat  = (X_flat - min_val) / denom
    X       = X_flat.reshape(X.shape)

    min_t   = min_val[:4]
    max_t   = max_val[:4]
    denom_t = max_t - min_t
    denom_t[denom_t == 0] = 1
    y       = ((y.reshape(-1, 4) - min_t) / denom_t).reshape(y.shape)

    idx  = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    split    = int(len(X) * 0.8)
    train_dl = DataLoader(TrafficDataset(X[:split], y[:split]), BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl   = DataLoader(TrafficDataset(X[split:], y[split:]), BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TrafficLSTM(
        input_size    = len(FEATURES),
        hidden_size   = HIDDEN,
        num_layers    = LAYERS,
        output_size   = len(TARGETS),
        pred_len      = PRED_LEN,
        dropout       = DROPOUT,
        bidirectional = True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[train] BiLSTM({LAYERS}x{HIDDEN}, bidirectional=True) | Params: {total_params:,}")
    print(f"[train] Device={device} | Epochs={EPOCHS} | Batch={BATCH_SIZE}")

    best_val   = float("inf")
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred   = model(xb)
            loss   = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()
        val_loss /= len(val_dl)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "lstm_best.pth"))
            print(f"  Epoch {epoch:3d} | train={train_loss:.4f} val={val_loss:.4f} lr={current_lr:.6f} ✅ saved")
        else:
            no_improve += 1
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d} | train={train_loss:.4f} val={val_loss:.4f} lr={current_lr:.6f} ({no_improve}/{PATIENCE})")

        if no_improve >= PATIENCE:
            print(f"[train] Early stopping at epoch {epoch}")
            break

    # Save scaler as plain dict
    scaler_data = {"min_": min_val, "max_": max_val}
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler_data, f)

    config = {
        "input_size":    len(FEATURES),
        "hidden_size":   HIDDEN,
        "num_layers":    LAYERS,
        "output_size":   len(TARGETS),
        "pred_len":      PRED_LEN,
        "dropout":       DROPOUT,
        "bidirectional": True,
        "features":      FEATURES,
    }
    with open(os.path.join(MODELS_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[train] ✅ Best val loss: {best_val:.4f}")
    print(f"[train] Saved → lstm/models/")


if __name__ == "__main__":
    train()