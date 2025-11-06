// app.js (ESM con imports)
// Ejecutá con: node app.js
// package.json debe tener: { "type": "module" }

import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { v4 as uuid } from 'uuid';
import { fileURLToPath } from 'url';
import cors from 'cors';

// ===== Paths base
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());

// Servir archivos estáticos
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));
app.use('/static', express.static(path.join(__dirname, 'static')));

// Rutas para servir los HTML
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'static', 'landing.html'));
});

app.get('/upload', (req, res) => {
  res.sendFile(path.join(__dirname, 'static', 'upload.html'));
});

const PORT = process.env.PORT || 3000;

// ===== Uploads (/uploads/edf y /uploads/models)
const uploadRoot = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadRoot)) fs.mkdirSync(uploadRoot, { recursive: true });

const storage = multer.diskStorage({
  destination: (_req, file, cb) => {
    const folder = path.join(uploadRoot, file.fieldname === 'model' ? 'models' : 'edf');
    fs.mkdirSync(folder, { recursive: true });
    cb(null, folder);
  },
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname || '');
    cb(null, `${uuid()}${ext}`);
  }
});

const upload = multer({ storage, limits: { fileSize: 200 * 1024 * 1024 } }); // 200MB

// ===== Utils
function absOrResolve(p) {
  if (!p) return null;
  return path.isAbsolute(p) ? p : path.resolve(__dirname, p);
}

function resolvePythonBin() {
  // Prioridad: .env -> venv comunes -> comandos en PATH
  const envBin = absOrResolve(process.env.PYTHON_BIN);
  const candidates = [
    envBin,
    path.join(__dirname, 'python/.venv/bin/python3'),
    path.join(__dirname, 'python/.venv/bin/python'),
    path.join(__dirname, 'python/.venv/Scripts/python.exe'), // Windows
    'python3',
    'python'
  ].filter(Boolean);

  for (const c of candidates) {
    // Si es ruta absoluta y existe, usarla; si es comando, probamos igual
    if (!path.isAbsolute(c) || fs.existsSync(c)) return c;
  }
  return 'python3';
}

// ===== Rutas
app.get('/health', (_req, res) => res.json({ ok: true }));

// POST /predict  (multipart/form-data)
// Campos:
//  - file  (EDF obligatorio)
//  - model (archivo .pt opcional; si no viene, usa DEFAULT_MODEL del .env)
app.post(
  '/predict',
  upload.fields([{ name: 'file', maxCount: 1 }, { name: 'model', maxCount: 1 }]),
  async (req, res) => {
    const edfUp = req.files?.file?.[0];
    if (!edfUp) return res.status(400).json({ ok: false, error: 'Falta archivo EDF (campo "file")' });

    const edfPath = path.resolve(edfUp.path);
    const modelPath =
      req.files?.model?.[0]?.path
        ? path.resolve(req.files.model[0].path)
        : absOrResolve(process.env.DEFAULT_MODEL);

    if (!modelPath || !fs.existsSync(modelPath)) {
      // Limpieza del EDF subido si falta el modelo
      fs.unlink(edfPath, () => {});
      return res.status(400).json({ ok: false, error: 'Modelo no provisto y DEFAULT_MODEL inválido/inexistente' });
    }

    const PRED_SCRIPT = absOrResolve(process.env.PRED_SCRIPT) || path.join(__dirname, 'python/predecir.py');
    try {
      const st = fs.statSync(PRED_SCRIPT);
      console.log('🧪 Script mtime:', st.mtime.toISOString(), 'size:', st.size);
    } catch (e) {
      console.log('🧪 Script stat error:', e?.message || e);
    }
    if (!fs.existsSync(PRED_SCRIPT)) {
      fs.unlink(edfPath, () => {});
      return res.status(500).json({ ok: false, error: `No se encontró el script: ${PRED_SCRIPT}` });
    }

    const PYTHON_BIN = resolvePythonBin();
    console.log('🧪 Python:', PYTHON_BIN);
    console.log('🧪 Script:', PRED_SCRIPT);
    console.log('🧪 EDF   :', edfPath);
    console.log('🧪 Model :', modelPath);

    const args = [PRED_SCRIPT, edfPath, modelPath, '--json', '--no-tuya']; // --json para que predecir.py imprima JSON y --no-tuya para deshabilitar Tuya

    let stdout = '';
    let stderr = '';

    const py = spawn(PYTHON_BIN, args, {
      cwd: __dirname, // raíz del proyecto
      env: { ...process.env, PYTHONUNBUFFERED: '1' }, // salida sin buffer
      stdio: ['ignore', 'pipe', 'pipe']
    });

    const killTimer = setTimeout(() => {
      try { py.kill('SIGKILL'); } catch {}
    }, Number(process.env.PY_TIMEOUT_MS || 120000)); // 120s default

    py.on('error', (err) => {
      clearTimeout(killTimer);
      fs.unlink(edfPath, () => {});
      return res.status(500).json({
        ok: false,
        error: 'No se pudo lanzar Python (spawn error)',
        details: String(err)
      });
    });

    py.stdout.on('data', (d) => (stdout += d.toString()));
    py.stderr.on('data', (d) => {
      const chunk = d.toString();
      stderr += chunk;
      console.error('[py stderr]', chunk.trim());
    });

    py.on('close', (code) => {
      clearTimeout(killTimer);
      // Limpio EDF subido (el modelo lo dejamos por si se reutiliza)
      fs.unlink(edfPath, () => {});

      if (code !== 0) {
        return res.status(500).json({
          ok: false,
          error: 'Python terminó con error',
          code,
          stderr: stderr.trim()
        });
      }

      try {
        // Extraer todos los posibles bloques JSON del stdout (viene mezclado con logs)
        const candidates = [];
        const text = stdout;

        // Heurística simple: tomar subcadenas que empiecen con "{" y tratar de parsearlas.
        for (let i = 0; i < text.length; i++) {
          if (text[i] !== '{') continue;
          for (let j = text.length; j > i; j--) {
            if (text[j - 1] !== '}') continue;
            const slice = text.slice(i, j);
            try {
              const obj = JSON.parse(slice);
              candidates.push(obj);
              break; // avanzar i a la próxima llave
            } catch (_) { /* probar con un cierre más chico */ }
          }
        }

        // Elegir el último candidato que parezca de predicción
        // - formato A: { file, prob, pred, gt, hit, X_shape }
        // - formato B: { ok, Archivo, Pred, Prob, GT, 'Forma X', ... }
        let chosen = null;
        for (let k = candidates.length - 1; k >= 0; k--) {
          const o = candidates[k];
          const hasA = ('pred' in o && ('prob' in o || 'Prob' in o));
          const hasB = ('Pred' in o && ('Prob' in o || 'prob' in o));
          if (hasA || hasB) { chosen = o; break; }
        }

        // Normalizar salida a un shape único para el front
        if (chosen) {
          // Calcular plot_url si hay plot_path
          const plotPath = chosen.plot_path ?? chosen.plotPath ?? null;
          let plot_url = null;
          if (plotPath) {
            try {
              const rel = path.relative(uploadRoot, plotPath);
              plot_url = `/uploads/${rel.replace(/\\/g, '/')}`;
            } catch (_) { /* noop */ }
          }
          // Calcular activations_url si hay activations_path
          const actPath = chosen.activations_path ?? chosen.activationsPath ?? null;
          let activations_url = null;
          if (actPath) {
            try {
              const relA = path.relative(uploadRoot, actPath);
              activations_url = `/uploads/${relA.replace(/\\/g, '/')}`;
            } catch (_) { /* noop */ }
          }
          // Opcional: URLs absolutas si PUBLIC_BASE_URL está presente
          const BASE = process.env.PUBLIC_BASE_URL?.replace(/\/+$/, '') || '';
          const abs_plot_url = plot_url ? `${BASE}${plot_url}` : null;
          const abs_activations_url = activations_url ? `${BASE}${activations_url}` : null;

          const norm = {
            ok: true,
            file: chosen.file ?? chosen.Archivo ?? null,
            gt: chosen.gt ?? chosen.GT ?? null,
            pred: chosen.pred ?? chosen.Pred ?? null,
            prob: typeof (chosen.prob ?? chosen.Prob) === 'number'
              ? (chosen.prob ?? chosen.Prob)
              : Number(chosen.prob ?? chosen.Prob),
            hit: ('hit' in chosen) ? chosen.hit : (('Hit' in chosen) ? chosen.Hit : null),
            x_shape: chosen.X_shape ?? chosen['Forma X'] ?? chosen.x_shape ?? null,
            plot_url,
            activations_url,
            abs_plot_url,
            abs_activations_url
          };
          return res.json(norm);
        }

        // Si no encontramos un JSON de predicción, devolver crudo con warning
        return res.status(200).json({
          ok: true,
          raw: stdout.trim(),
          warning: 'No se pudo extraer JSON de predicción; devolviendo salida cruda'
        });
      } catch (e) {
        return res.status(200).json({
          ok: true,
          raw: stdout.trim(),
          warning: `Error al extraer predicción: ${e.message}`
        });
      }
    });
  }
);

app.listen(PORT, () => {
  console.log(`🚀 Servidor escuchando en http://localhost:${PORT}`);
});