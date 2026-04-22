# lstm/predict.py (BiLSTM)
import os, json, pickle, threading
import numpy as np
import torch
import torch.nn as nn

LSTM_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(LSTM_DIR, "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "lstm_best.pth")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
CONFIG_PATH = os.path.join(MODELS_DIR, "config.json")

SEQ_LEN  = 60
PRED_LEN = 30

_load_lock = threading.Lock()


class TrafficLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 output_size, pred_len, dropout, bidirectional=True):
        super().__init__()
        self.pred_len      = pred_len
        self.output_size   = output_size
        self.bidirectional = bidirectional
        directions         = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0,
            bidirectional = bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size * directions, output_size * pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1, :])
        out    = self.fc(out)
        return out.view(-1, self.pred_len, self.output_size)


_model    = None
_scaler   = None
_config   = None
_features = None


def _load():
    global _model, _scaler, _config, _features

    # FIX: guard against concurrent calls before model is initialised
    with _load_lock:
        if _model is not None:
            return  # another thread already loaded while we waited

        for p in [MODEL_PATH, SCALER_PATH, CONFIG_PATH]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing: {p}\nRun: python lstm/train_lstm.py")

        with open(CONFIG_PATH) as f:
            _config = json.load(f)

        # FIX: always derive feature list from config — no hardcoded fallback
        # The config is written by train_lstm.py and reflects whatever FEATURES
        # was used at training time (currently 7: includes avg_waiting).
        _features = _config["features"]

        with open(SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)

        _model = TrafficLSTM(
            input_size    = _config["input_size"],
            hidden_size   = _config["hidden_size"],
            num_layers    = _config["num_layers"],
            output_size   = _config["output_size"],
            pred_len      = _config["pred_len"],
            dropout       = _config["dropout"],
            bidirectional = _config.get("bidirectional", True),
        )
        _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        _model.eval()
        print(f"[LSTM] BiLSTM model loaded  (features={_features})")


def is_ready() -> bool:
    return all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, CONFIG_PATH])


def predict(history: list) -> dict:
    global _model, _scaler, _features
    if _model is None:
        _load()

    # FIX: feature list comes entirely from config — no stale fallback
    features = _features
    min_val  = _scaler["min_"]
    max_val  = _scaler["max_"]
    denom    = max_val - min_val
    denom[denom == 0] = 1

    recent = list(history[-SEQ_LEN:])
    while len(recent) < SEQ_LEN:
        recent.insert(0, {f: 0.0 for f in features})

    arr        = np.array([[r.get(f, 0.0) for f in features] for r in recent], dtype=np.float32)
    arr_scaled = (arr - min_val) / denom
    x          = torch.FloatTensor(arr_scaled).unsqueeze(0)

    with torch.no_grad():
        pred = _model(x).squeeze(0).numpy()

    min_t = min_val[:4]
    max_t = max_val[:4]
    dt    = max_t - min_t
    dt[dt == 0] = 1
    pred  = np.maximum(pred * dt + min_t, 0).round().astype(int)

    return {
        "north":    pred[:, 0].tolist(),
        "south":    pred[:, 1].tolist(),
        "east":     pred[:, 2].tolist(),
        "west":     pred[:, 3].tolist(),
        "next_30s": {
            "north": int(pred[:, 0].mean()),
            "south": int(pred[:, 1].mean()),
            "east":  int(pred[:, 2].mean()),
            "west":  int(pred[:, 3].mean()),
        }
    }