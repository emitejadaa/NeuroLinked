import time, uuid, hashlib, hmac, json, requests, sys, os, re, torch, numpy as np, mne
import matplotlib
matplotlib.use('Agg')  # backend sin GUI para servidores
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import torch.nn as nn
import torch.nn.functional as F

# ===============================
# CONFIGURACIÓN TUYA FIJA
# ===============================
CLIENT_ID     = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
ACCESS_TOKEN  = os.getenv("ACCESS_TOKEN")
DEVICE_ID     = os.getenv("DEVICE_ID")
HOST          = os.getenv("HOST")

# ===============================
# FUNCIÓN PARA ACCIONAR TUYA
# ===============================
def tuya_switch(value: int | bool):
    body = {"commands": [{"code": "switch_led", "value": bool(value)}]}
    t = str(int(time.time() * 1000))
    nonce = str(uuid.uuid4())

    body_json = json.dumps(body, separators=(",", ":")).encode("utf-8")
    content_sha256 = hashlib.sha256(body_json).hexdigest()

    optional_signature_key = ""
    path = f"/v1.0/devices/{DEVICE_ID}/commands"
    url_for_sign = path

    string_to_sign = f"POST\n{content_sha256}\n{optional_signature_key}\n{url_for_sign}"
    to_sign = CLIENT_ID + ACCESS_TOKEN + t + nonce + string_to_sign

    sign = hmac.new(
        CLIENT_SECRET.encode("utf-8"),
        to_sign.encode("utf-8"),
        hashlib.sha256
    ).hexdigest().upper()

    headers = {
        "client_id": CLIENT_ID,
        "access_token": ACCESS_TOKEN,
        "t": t,
        "sign": sign,
        "sign_method": "HMAC-SHA256",
        "nonce": nonce,
        "Content-Type": "application/json",
    }

    resp = requests.post(f"{HOST}{path}", headers=headers, data=body_json, timeout=20)
    print("💡 Tuya Status:", resp.status_code, resp.text)


# ===============================
# MODELO Y PREDICCIÓN
# ===============================
def predict_one_edf(
    edf_path: str,
    model_path: str,
    *,
    classes: list[str] = ["rest", "left"],
    n_channels: int = 32,
    T: int = 513,
    fs: float = 256.0,
    tmin: float = 0.0,
    bandpass: tuple[float, float] | None = (1.0, 40.0),
    notch: float | None = 50.0,
    scale_to_mV: bool = True,
    threshold: float = 0.5,
) -> dict:

    class PositionalEncoding(nn.Module):
        def __init__(self, num_hiddens, dropout, max_len=1000):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.p = torch.zeros((1, max_len, num_hiddens))
            x = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
                10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
            )
            self.p[:, :, 0::2] = torch.sin(x)
            self.p[:, :, 1::2] = torch.cos(x)
        def forward(self, x):
            x = x + self.p[:, :x.shape[1], :].to(x.device)
            return self.dropout(x)

    class TransformerBlock(nn.Module):
        def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1):
            super().__init__()
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
            self.ff = nn.Sequential(
                nn.Linear(embed_dim, dim_feedforward),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, embed_dim),
            )
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.dropout = dropout
        def forward(self, x):
            y, _ = self.attn(x, x, x)
            x = self.norm1(x + F.dropout(y, self.dropout, training=self.training))
            y = self.ff(x)
            return self.norm2(x + F.dropout(y, self.dropout, training=self.training))

    class EEGModel(nn.Module):
        def __init__(self, C, dropout=0.125):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(C, C, 11, padding=5, bias=False),
                nn.BatchNorm1d(C),
                nn.ReLU(),
                nn.Conv1d(C, C * 2, 11, padding=5, bias=False),
                nn.BatchNorm1d(C * 2),
            )
            self.trans = nn.Sequential(
                PositionalEncoding(C * 2, dropout),
                TransformerBlock(C * 2, 4, max(1, C // 8), dropout),
                TransformerBlock(C * 2, 4, max(1, C // 8), dropout),
            )
            self.mlp = nn.Sequential(
                nn.Linear(C * 2, max(1, C // 2)),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(max(1, C // 2), 1),
            )
        def forward(self, x):
            x = self.conv(x)
            x = self.trans(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = x.mean(dim=-1)
            return self.mlp(x)

    def _ensure_T(X, exp_t):
        cur = X.shape[-1]
        if cur == exp_t:
            return X
        if abs(cur - exp_t) <= 2:
            return X[..., :exp_t] if cur > exp_t else np.pad(X, [(0,0),(0,0),(0,exp_t-cur)], "constant")
        raise ValueError(f"T={cur} != {exp_t}")

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    eeg_inds = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, exclude='bads')
    raw.pick(eeg_inds[:n_channels])

    if bandpass:
        raw.filter(*bandpass, verbose="ERROR")
    if notch and notch > 0:
        raw.notch_filter(notch, verbose="ERROR")
    if abs(raw.info["sfreq"] - fs) > 1e-6:
        raw.resample(fs, npad="auto", verbose="ERROR")

    sf = raw.info["sfreq"]
    tmax_eff = tmin + (T - 1) / sf

    X = None
    try:
        events, _ = mne.events_from_annotations(raw)
        if len(events)>0:
            ep = mne.Epochs(raw, events, tmin=tmin, tmax=tmax_eff, baseline=None, preload=True, verbose="ERROR")
            if len(ep)>0:
                X = ep.get_data()
    except Exception:
        pass

    if X is None or X.shape[0]==0:
        start = int(tmin*sf); stop = start + T
        data = raw.get_data()[:n_channels, start:stop]
        X = data[None,...]
    X = (X*1e3 if scale_to_mV else X).astype(np.float32)
    X = _ensure_T(X, T)

    # === Helpers for labels (moved up so plotting can use them) ===
    def _norm_label(txt):
        if not isinstance(txt, str): txt = str(txt)
        for k, v in {"rest": ["rest", "t0", "T0"], "left": ["left", "t1", "T1"], "right": ["right", "t2", "T2"]}.items():
            if txt.lower() in [x.lower() for x in v]:
                return k
        return None

    def label_from_edf(path):
        try:
            raw0 = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
            if raw0.annotations and len(raw0.annotations) > 0:
                lab = _norm_label(raw0.annotations[0]["description"])
                if lab:
                    return lab
        except Exception:
            pass
        m = re.search(r"(left|right|rest|t0|t1|t2)", os.path.basename(path), re.I)
        return _norm_label(m.group(1)) if m else None

    # === Build model & predict BEFORE plotting (so we can put Pred in the title) ===
    model = EEGModel(n_channels)
    obj = torch.load(model_path, map_location="cpu")
    if isinstance(obj, dict):
        sd = obj.get("state_dict", obj)
        model.load_state_dict({k.replace("model.", "").replace("module.", ""): v for k, v in sd.items()}, strict=False)
    model.eval()

    with torch.no_grad():
        out = model(torch.from_numpy(X)).squeeze(-1).numpy()
        prob = 1 / (1 + np.exp(-out))[0]
        pred = classes[int(prob > threshold)]

    # === Plot EXACTLY like the reference snippet ===
    plot_path = None
    try:
        plt.figure()  # fresh figure with default size/style
        # sample: take one epoch (first) and plot channels along columns
        sample = X[0, :n_channels, :]  # (C, T)
        # Title values
        PREDICTED = pred
        ACTUAL = label_from_edf(edf_path) or "?"

        # Prepare output path
        uploads_dir = os.path.dirname(os.path.dirname(edf_path))  # .../uploads
        plots_dir = os.path.join(uploads_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(edf_path))[0]
        plot_path = os.path.join(plots_dir, f"{base_name}.png")

        # Plot with the same API calls/order as the provided code
        plt.plot(sample.T)
        plt.title(
            "Exemplar of epoched data, for electrode 0-63\\n"
            f"Actual Label : {ACTUAL}\\n"
            f"Predicted Label : {PREDICTED}"
        )
        plt.ylabel("V")
        plt.xlabel("Epoched Sample")
        # In headless mode we save instead of show
        plt.savefig(plot_path)
        plt.clf()
        plt.close()
    except Exception as _e:
        plot_path = None

    gt = label_from_edf(edf_path)
    hit = (pred == gt) if gt in classes else None

    return {"file": edf_path, "prob": float(prob), "pred": pred, "gt": gt, "hit": hit, "X_shape": X.shape, "plot_path": plot_path}


# ===============================
# MAIN: SOLO RUTA LOCAL (con opción de model path)
# ===============================
# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    import argparse

    def human_lines(res: dict) -> list[str]:
        """Genera las mismas líneas 'humanas' que imprimías antes."""
        return [
            f"Archivo: {res.get('file')}",
            f"GT: {res.get('gt')}",
            f"Pred: {res.get('pred')}",
            f"Prob: {res.get('prob'):.4f}" if isinstance(res.get('prob'), (float, int)) else f"Prob: {res.get('prob')}",
            f"Hit: {res.get('hit')}",
            f"Forma X: {tuple(res.get('X_shape')) if res.get('X_shape') is not None else None}",
        ]

    ap = argparse.ArgumentParser()
    ap.add_argument("edf", help="Ruta local del EDF")
    ap.add_argument("model", nargs="?", default="python/left_rest_model_20251022_142504.pt",
                    help="Ruta local del modelo .pt (opcional)")
    ap.add_argument("--no-tuya", action="store_true", help="No accionar Tuya")
    ap.add_argument("--json", action="store_true", help="Imprimir JSON (para el front)")
    args = ap.parse_args()

    EDF_PATH = args.edf
    MODEL_PATH = args.model

    if not os.path.exists(EDF_PATH):
        print(json.dumps({"ok": False, "error": f"No existe el EDF: {EDF_PATH}"}))
        sys.exit(2)

    if not os.path.exists(MODEL_PATH):
        print(json.dumps({"ok": False, "error": f"No existe el modelo: {MODEL_PATH}"}))
        sys.exit(3)

    try:
        res = predict_one_edf(EDF_PATH, MODEL_PATH)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"Fallo en predict_one_edf: {e}"}))
        sys.exit(4)

    # Acción Tuya (opcional)
    tuya_status = None
    if not args.no_tuya:
        try:
            tuya_switch(0 if res["pred"] == "rest" else 1)
            tuya_status = "sent"
        except Exception as e:
            tuya_status = f"error: {e}"

    # --- Salidas ---
    # 1) JSON (para el front) -> usa las claves con los mismos nombres "humanos"
    out_json = {
        "ok": True,
        "Archivo": res.get("file"),
        "GT": res.get("gt"),
        "Pred": res.get("pred"),
        "Prob": float(res.get("prob")) if res.get("prob") is not None else None,
        "Hit": res.get("hit"),
        "Forma X": list(res.get("X_shape")) if res.get("X_shape") is not None else None,
        "plot_path": res.get("plot_path"),
        "tuya_status": tuya_status,
    }

    if args.json:
        # JSON limpio para que Node lo parsee directo y lo devuelva al front
        print(json.dumps(out_json))
    else:
        # 2) Texto humano (igual que antes) + status Tuya
        print("")  # salto de línea como antes
        for line in human_lines(res):
            print(line)
        if tuya_status is not None:
            print(f"Tuya: {tuya_status}")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("edf")
    ap.add_argument("model", nargs="?", default="/Users/bensagra/Downloads/left_rest_model_20251022_142504.pt")
    ap.add_argument("--no-tuya", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    EDF_PATH = args.edf
    MODEL_PATH = args.model

    res = predict_one_edf(EDF_PATH, MODEL_PATH)

    if args.json:
        print(json.dumps(res))  # JSON limpio para Node
    else:
        # tu salida original:
        print(f"\nArchivo: {res['file']}")
        print(f"GT: {res['gt']}")
        print(f"Pred: {res['pred']}")
        print(f"Prob: {res['prob']:.4f}")
        print(f"Hit: {res['hit']}")
        print(f"Forma X: {res['X_shape']}")

    if not args.no_tuya:
        tuya_switch(0 if res["pred"] == "rest" else 1)
    if len(sys.argv) < 2:
        print("Uso: python predecir.py <RUTA_LOCAL_DEL_EDF> [RUTA_LOCAL_DEL_MODELO_PT]")
        sys.exit(1)

    EDF_PATH = sys.argv[1]
    if not os.path.exists(EDF_PATH):
        print(f"Error: no existe el archivo local: {EDF_PATH}")
        sys.exit(2)

    # Path del modelo: usa el default si no se pasa por consola
    MODEL_PATH = sys.argv[2] if len(sys.argv) >= 3 else "python/left_rest_model_20251022_142504.pt"
    if not os.path.exists(MODEL_PATH):
        print(f"Error: no existe el modelo: {MODEL_PATH}")
        sys.exit(3)

    # Ejecutar predicción
    res = predict_one_edf(EDF_PATH, MODEL_PATH)

    # Mostrar resultados
    print(f"\nArchivo: {res['file']}")
    print(f"GT: {res['gt']}")
    print(f"Pred: {res['pred']}")
    print(f"Prob: {res['prob']:.4f}")
    print(f"Hit: {res['hit']}")
    print(f"Forma X: {res['X_shape']}")

    # Accionar Tuya según predicción (rest -> OFF, otro -> ON)
    if res["pred"] == "rest":
        tuya_switch(0)
    else:
        tuya_switch(1)