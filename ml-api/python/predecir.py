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

    try:
        resp = requests.post(f"{HOST}{path}", headers=headers, data=body_json, timeout=20)
        txt = resp.text
        ok = resp.status_code in (200, 201)
        parsed = None
        try:
            parsed = resp.json()
            if isinstance(parsed, dict) and "success" in parsed:
                ok = ok and bool(parsed.get("success"))
        except Exception:
            parsed = None
        print("💡 Tuya Status:", resp.status_code, txt)
        return ok, {"status_code": resp.status_code, "text": txt, "json": parsed}
    except Exception as e:
        print("💥 Tuya error:", e)
        return False, {"error": str(e)}


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

    # ---- Captura de activaciones para graficar "neuronas que se prenden" ----
    ACTIVS = {}

    def _hook_save(name):
        def _fn(module, inputs, output):
            try:
                # Guardamos en CPU y como numpy cuando sea posible
                if isinstance(output, tuple):
                    # Para MultiheadAttention: (attn_out, attn_weights)
                    ACTIVS[name] = tuple(
                        (o.detach().cpu().numpy() if hasattr(o, "detach") else o) for o in output
                    )
                else:
                    ACTIVS[name] = output.detach().cpu().numpy() if hasattr(output, "detach") else output
            except Exception:
                ACTIVS[name] = output
        return _fn

    # Última capa de la pila conv (BatchNorm1d) -> feature map (B, C*2, T)
    _ = model.conv[-1].register_forward_hook(_hook_save("conv_out"))

    # Bloques Transformer (en la secuencia trans: [PositionalEncoding, Block1, Block2])
    _ = model.trans[1].register_forward_hook(_hook_save("trans1_out"))
    _ = model.trans[2].register_forward_hook(_hook_save("trans2_out"))

    # Atención interna: registramos hook en los MultiheadAttention para captar pesos
    _ = model.trans[1].attn.register_forward_hook(_hook_save("attn1_out_and_weights"))
    _ = model.trans[2].attn.register_forward_hook(_hook_save("attn2_out_and_weights"))

    # Antes de la MLP: queremos el vector (B, C*2) que entra a la primera Linear
    def _mlp_in_hook(module, inputs, output):
        # inputs es una tupla; tomamos el primer tensor
        x0 = inputs[0]
        ACTIVS["mlp_in"] = x0.detach().cpu().numpy() if hasattr(x0, "detach") else x0
    _ = model.mlp[0].register_forward_hook(_mlp_in_hook)

    # Activación de la capa oculta (salida de la ReLU)
    _ = model.mlp[1].register_forward_hook(_hook_save("mlp_h"))

    # Logit final (B, 1)
    _ = model.mlp[-1].register_forward_hook(_hook_save("mlp_logits"))

    obj = torch.load(model_path, map_location="cpu")
    if isinstance(obj, dict):
        sd = obj.get("state_dict", obj)
        model.load_state_dict({k.replace("model.", "").replace("module.", ""): v for k, v in sd.items()}, strict=False)
    model.eval()

    with torch.no_grad():
        out = model(torch.from_numpy(X)).squeeze(-1).numpy()
        prob = 1 / (1 + np.exp(-out))[0]
        pred = classes[int(prob > threshold)]

    # ---- Gráfico 2: Top activaciones de neuronas (entrada a la MLP) ----
    activations_plot_path = None
    activations_url = None
    try:
        mlp_in = ACTIVS.get("mlp_in", None)
        if mlp_in is not None:
            v = mlp_in[0]  # (C*2,)
            # Top-K por magnitud
            k = min(20, v.shape[-1])
            idx = np.argsort(np.abs(v))[-k:][::-1]
            top_vals = v[idx]
            top_idx = idx

            # === Preparar pesos/canales y top ocultas (requeridos por el grafo) ===
            W1 = None
            W2 = None
            try:
                W1 = model.mlp[0].weight.detach().cpu().numpy()  # (H, C*2)
                W2 = model.mlp[-1].weight.detach().cpu().numpy() # (1, H)
            except Exception:
                pass

            h = ACTIVS.get("mlp_h", None)  # salida de ReLU, shape (1, H)
            if h is not None and hasattr(h, "shape"):
                h_vec = np.asarray(h)[0]  # (H,)
            else:
                # Si no capturamos, estimamos como ReLU(W1 @ v)
                Hsize = W1.shape[0] if W1 is not None else max(1, v.shape[-1]//2)
                h_vec = np.maximum(0.0, (W1 @ v) if W1 is not None else np.zeros((Hsize,), dtype=float))

            # Tamaños de subconjuntos para el dibujo
            k_in = len(top_idx)                           # cantidad de inputs a mostrar (Top-K de mlp_in)
            m_h = int(min(12, h_vec.shape[0]))           # hasta 12 neuronas ocultas
            hid_idx = np.argsort(np.abs(h_vec))[-m_h:][::-1]
            h_top = h_vec[hid_idx]

            uploads_dir = os.path.dirname(os.path.dirname(edf_path))  # .../uploads
            plots_dir = os.path.join(uploads_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(edf_path))[0]
            activations_plot_path = os.path.join(plots_dir, f"{base_name}_activations.png")

            # --- Visualización tipo red por capas (Input -> Hidden -> Output) ---
            # Estilo y helpers
            import matplotlib as mpl

            # Colormap y normalizadores
            cmap = mpl.colormaps.get_cmap('viridis')
            def _to_rgba01(x):
                x = float(np.clip(x, 0.0, 1.0))
                r, g, b, a = cmap(x)
                return (r, g, b, 1.0)

            # (el resto del código permanece igual)

            # Layout: tres columnas X = {0,1,2}
            x_in, x_h, x_out = 0.0, 1.0, 2.0
            # Y distribuciones verticales
            y_in = np.linspace(0.0, 1.0, k_in)
            y_h  = np.linspace(0.0, 1.0, m_h)
            y_out = np.array([0.7, 0.3])  # up=left, down=rest

            # Normalizaciones para estilos (0..1)
            def _norm01(arr):
                arr = np.abs(np.asarray(arr).astype(float))
                if arr.size == 0:
                    return arr
                a, b = float(arr.min()), float(arr.max())
                rng = (b - a) if (b - a) > 1e-12 else 1.0
                return (arr - a) / rng

            in_strength = _norm01(top_vals)                  # para nodos de entrada
            h_strength  = _norm01(h_top)                     # para nodos ocultos
            # Contribución de input->hidden usando |w_ij * v_i|
            if W1 is not None:
                W1_sub = W1[hid_idx, :][:, top_idx]          # (m_h, k_in)
                contrib_in_h = np.abs(W1_sub * top_vals[None, :])
                contrib_in_h = contrib_in_h / (contrib_in_h.max() + 1e-9)
            else:
                contrib_in_h = np.zeros((m_h, k_in), dtype=float)

            # Contribución de hidden->output usando |w2_j * h_j|
            if W2 is not None:
                w2 = W2[0, hid_idx]                          # (m_h,)
                contrib_h_out = np.abs(w2 * h_top)
                c_ho = contrib_h_out / (contrib_h_out.max() + 1e-9)
            else:
                c_ho = np.zeros((m_h,), dtype=float)

            # --- Figura con estética clara y color por intensidad ---
            fig = plt.figure(figsize=(9, 6))
            ax = fig.add_subplot(111)
            # Fondo oscuro para resaltar intensidades
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            ax.set_axis_off()
            ax.set_xlim(-0.25, 2.7)
            ax.set_ylim(-0.1, 1.1)

            # Sutilezas tipográficas en claro
            title_color = '#e6edf3'
            text_color  = '#e6edf3'
            edge_color  = '#e6edf3'

            # --- Dibujar conexiones Input -> Hidden con "glow" + color por contribución ---
            for j in range(m_h):
                for i in range(k_in):
                    c = float(contrib_in_h[j, i])
                    col = _to_rgba01(c)
                    # Glow (trazo grueso translúcido)
                    ax.plot([x_in, x_h], [y_in[i], y_h[j]], linewidth=3.8, alpha=0.08, color='w', zorder=1)
                    # Trazo principal coloreado por contribución
                    ax.plot([x_in, x_h], [y_in[i], y_h[j]], linewidth=1.6 + 2.2*c, alpha=0.8*c + 0.2, color=col, zorder=2)

            # --- Dibujar conexiones Hidden -> Output con "glow" + color por contribución ---
            for j in range(m_h):
                c = float(c_ho[j])
                col = _to_rgba01(c)
                # H->Left
                ax.plot([x_h, x_out], [y_h[j], y_out[0]], linewidth=4.4, alpha=0.08, color='w', zorder=1)
                ax.plot([x_h, x_out], [y_h[j], y_out[0]], linewidth=1.8 + 2.6*c, alpha=0.85*c + 0.15, color=col, zorder=2)
                # H->Rest (más tenue)
                ax.plot([x_h, x_out], [y_h[j], y_out[1]], linewidth=3.2, alpha=0.05, color='w', zorder=1)
                ax.plot([x_h, x_out], [y_h[j], y_out[1]], linewidth=1.2 + 1.8*c, alpha=0.6*c + 0.15, color=col, zorder=2)

            # --- Nodos ---
            # Tamaños escalan al cuadrado de la intensidad para más contraste
            sizes_in = 60.0 + 500.0 * (in_strength ** 2)
            sizes_h  = 70.0 + 520.0 * (h_strength ** 2)
            # Colores por intensidad
            cols_in = [ _to_rgba01(v) for v in in_strength ]
            cols_h  = [ _to_rgba01(v) for v in h_strength ]

            # Input nodes
            ax.scatter(np.full(k_in, x_in), y_in, s=sizes_in, c=cols_in, edgecolors=edge_color, linewidths=0.6, zorder=3)
            # Hidden nodes
            ax.scatter(np.full(m_h, x_h),  y_h,  s=sizes_h,  c=cols_h,  edgecolors=edge_color, linewidths=0.6, zorder=3)

            # Output nodes (dos) coloreados por probabilidad
            p_left = float(prob); p_rest = float(1.0 - prob)
            size_left = 220.0 + 520.0 * (p_left ** 2)
            size_rest = 220.0 + 520.0 * (p_rest ** 2)
            col_left = _to_rgba01(p_left)
            col_rest = _to_rgba01(p_rest)

            ax.scatter([x_out], [y_out[0]], s=size_left, c=[col_left], edgecolors=edge_color, linewidths=0.8, zorder=4)
            ax.scatter([x_out], [y_out[1]], s=size_rest, c=[col_rest], edgecolors=edge_color, linewidths=0.8, zorder=4)

            # Etiquetas de capas
            ax.text(x_in, 1.055, "Input (Top-K)", ha='center', va='bottom', fontsize=10, color=text_color)
            ax.text(x_h,  1.055, f"Hidden (m={m_h})", ha='center', va='bottom', fontsize=10, color=text_color)
            ax.text(x_out,1.055, "Output (Prob)", ha='center', va='bottom', fontsize=10, color=text_color)

            # Etiquetas de clases con sus probabilidades (más grandes y claras)
            ax.text(x_out + 0.18, y_out[0], f"left: {p_left:.2f}", va='center', fontsize=12, weight='bold', color=text_color)
            ax.text(x_out + 0.18, y_out[1], f"rest: {p_rest:.2f}", va='center', fontsize=12, weight='bold', color=text_color)

            # Título con predicción
            ax.set_title("Neuron graph — intensity = activation / contribution", color=title_color, fontsize=12, pad=10)

            # Pequeña barra de referencia (colorbar manual) en la esquina inferior izquierda
            # para indicar que más claro = mayor activación/contribución
            cb_x0, cb_y0, cb_w, cb_h = -0.15, -0.06, 0.25, 0.035
            for i_cb in range(200):
                t_cb = i_cb / 199.0
                ax.add_patch(plt.Rectangle((cb_x0 + cb_w * t_cb, cb_y0), cb_w / 200.0, cb_h,
                                           facecolor=_to_rgba01(t_cb), edgecolor='none', zorder=0))
            ax.text(cb_x0, cb_y0 + cb_h + 0.005, "low", fontsize=8, color=text_color, ha='left', va='bottom')
            ax.text(cb_x0 + cb_w, cb_y0 + cb_h + 0.005, "high", fontsize=8, color=text_color, ha='right', va='bottom')

            plt.tight_layout()
            plt.savefig(activations_plot_path, dpi=240, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.clf(); plt.close()

            # Construir URL relativa para el front (Express sirve /uploads como estático)
            activations_url = None
            try:
                uploads_dir_abs = os.path.dirname(os.path.dirname(edf_path))  # .../uploads
                rel_from_uploads = os.path.relpath(activations_plot_path, uploads_dir_abs).replace(os.sep, "/")
                activations_url = f"/uploads/{rel_from_uploads}"
            except Exception:
                activations_url = None
    except Exception:
        activations_plot_path = None
        activations_url = None

    # === Plot EXACTLY like the reference snippet ===
    plot_path = None
    plot_url = None
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
        try:
            uploads_dir_abs = os.path.dirname(os.path.dirname(edf_path))  # .../uploads
            rel_from_uploads = os.path.relpath(plot_path, uploads_dir_abs).replace(os.sep, "/")
            plot_url = f"/uploads/{rel_from_uploads}"
        except Exception:
            plot_url = None
    except Exception as _e:
        plot_path = None
        plot_url = None

    gt = label_from_edf(edf_path)
    hit = (pred == gt) if gt in classes else None
    
    return {
        "file": edf_path,
        "prob": float(prob),
        "pred": pred,
        "gt": gt,
        "hit": hit,
        "X_shape": X.shape,
        "plot_path": plot_path,
        "plot_url": plot_url,
        "activations_path": activations_plot_path,
        "activations_url": activations_url,
    }


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
            f"Activations plot: {res.get('activations_path')}",
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
        required_envs = {
            "CLIENT_ID": CLIENT_ID,
            "CLIENT_SECRET": CLIENT_SECRET,
            "ACCESS_TOKEN": ACCESS_TOKEN,
            "DEVICE_ID": DEVICE_ID,
            "HOST": HOST,
        }
        missing = [k for k, v in required_envs.items() if not v]
        if missing:
            tuya_status = {"sent": False, "ok": False, "reason": f"missing env vars: {', '.join(missing)}"}
            sys.stderr.write(f"Tuya skipped, missing env: {', '.join(missing)}\n")
            sys.stderr.flush()
        else:
            try:
                desired = 0 if res["pred"] == "rest" else 1
                ok, info = tuya_switch(desired)
                tuya_status = {"sent": True, "ok": bool(ok), "info": info, "desired": desired}
                sys.stderr.write(f"Tuya sent -> ok={ok}, desired={desired}\n")
                sys.stderr.flush()
            except Exception as e:
                tuya_status = {"sent": True, "ok": False, "error": str(e)}
                sys.stderr.write(f"Tuya error: {e}\n")
                sys.stderr.flush()

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
        "plot_url": res.get("plot_url"),
        "activations_path": res.get("activations_path"),
        "activations_url": res.get("activations_url"),
        "tuya_status": tuya_status,
    }

    # Emitimos SIEMPRE JSON a STDOUT (para el front)
    sys.stdout.write(json.dumps(out_json))
    sys.stdout.flush()

    # Opcional: cuando no se pide --json, además mandamos resumen humano a STDERR para logging
    if not args.json:
        sys.stderr.write("\n")  # salto de línea
        for line in human_lines(res):
            sys.stderr.write(line + "\n")
        if tuya_status is not None:
            sys.stderr.write(f"Tuya: {tuya_status}\n")
        sys.stderr.flush()