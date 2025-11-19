# 🔧 Corrección Crítica: Alineación de Preprocesamiento EEG

## ⚠️ PROBLEMA IDENTIFICADO

El script de predicción (`predecir.py`) tenía **parámetros de preprocesamiento inconsistentes** con los usados durante el entrenamiento del modelo, causando predicciones incorrectas o aleatorias.

## 📊 Comparación: Entrenamiento vs Predicción Original

### Durante el Entrenamiento (notebook)

```python
# Parámetros del dataset y preprocesamiento
SAMPLE_RATE = 160  # Hz
DURATION = 3.0  # segundos
TARGET_LENGTH = 480  # muestras (160 * 3.0)
CANALES_EEG = 64  # todos los canales EEG disponibles
CLASES = ["left", "right"]  # solo left y right

# Preprocesamiento:
- ✅ Frecuencia: 160 Hz (nativa del dataset)
- ✅ Duración: 3 segundos = 480 muestras
- ✅ Canales: 64 (todos los EEG disponibles)
- ✅ Filtros: NINGUNO (sin bandpass, sin notch)
- ✅ Escala: Convertir a mV (× 1e3)
- ✅ Normalización de longitud: padding con ceros si es necesario
```

### En Predicción (ANTES - ❌ INCORRECTO)

```python
# Parámetros por defecto INCORRECTOS
fs = 256.0  # Hz ❌ DIFERENTE!
T = 513  # muestras ❌ DIFERENTE!
n_channels = 32  # ❌ DIFERENTE!
classes = ["rest", "left"]  # ❌ DIFERENTE!
bandpass = (1.0, 40.0)  # ❌ FILTRO NO USADO EN ENTRENAMIENTO!
notch = 50.0  # ❌ FILTRO NO USADO EN ENTRENAMIENTO!
```

**Resultado:** Las señales EEG se procesaban de forma completamente diferente, generando features incompatibles con el modelo entrenado. 🚨

## ✅ SOLUCIÓN APLICADA

### En Predicción (DESPUÉS - ✅ CORRECTO)

```python
# Parámetros alineados con el entrenamiento
classes = ["left", "right"]  # ✅ Coincide
n_channels = 64  # ✅ Coincide
T = 480  # ✅ Coincide (160Hz * 3.0s)
fs = 160.0  # ✅ Coincide
bandpass = None  # ✅ Sin filtro (como en entrenamiento)
notch = None  # ✅ Sin notch (como en entrenamiento)
scale_to_mV = True  # ✅ Coincide
```

## 🔍 Cambios Específicos en `predecir.py`

### Líneas 73-85 (función `predict_one_edf`)

**ANTES:**
```python
def predict_one_edf(
    edf_path: str,
    model_path: str,
    *,
    classes: list[str] = ["rest", "left"],  # ❌
    n_channels: int = 32,  # ❌
    T: int = 513,  # ❌
    fs: float = 256.0,  # ❌
    tmin: float = 0.0,
    bandpass: tuple[float, float] | None = (1.0, 40.0),  # ❌
    notch: float | None = 50.0,  # ❌
    scale_to_mV: bool = True,
    threshold: float = 0.5,
) -> dict:
```

**DESPUÉS:**
```python
def predict_one_edf(
    edf_path: str,
    model_path: str,
    *,
    classes: list[str] = ["left", "right"],  # ✅ Correcto
    n_channels: int = 64,  # ✅ Correcto
    T: int = 480,  # ✅ Correcto (160Hz * 3.0s)
    fs: float = 160.0,  # ✅ Correcto
    tmin: float = 0.0,
    bandpass: tuple[float, float] | None = None,  # ✅ Sin filtro
    notch: float | None = None,  # ✅ Sin notch
    scale_to_mV: bool = True,
    threshold: float = 0.5,
) -> dict:
```

## 🧪 Validación del Pipeline de Preprocesamiento

### Flujo Correcto Ahora:

1. **Carga del EDF**
   ```python
   raw = mne.io.read_raw_edf(edf_path, preload=True)
   eeg_inds = mne.pick_types(raw.info, eeg=True)
   raw.pick(eeg_inds[:64])  # ✅ 64 canales
   ```

2. **Filtrado** (DESHABILITADO - correcto)
   ```python
   if bandpass:  # None -> se salta
       raw.filter(*bandpass)
   if notch and notch > 0:  # None -> se salta
       raw.notch_filter(notch)
   ```

3. **Resampling**
   ```python
   if abs(raw.info["sfreq"] - 160.0) > 1e-6:  # ✅ 160Hz
       raw.resample(160.0)
   ```

4. **Extracción de épocas**
   ```python
   tmax_eff = 0.0 + (480 - 1) / 160.0  # ✅ 3 segundos
   # Intenta con eventos, si no usa ventana fija de 480 muestras
   ```

5. **Normalización de longitud**
   ```python
   def _ensure_T(X, exp_t=480):  # ✅ 480 muestras
       if cur < exp_t:
           return np.pad(X, [...], "constant")  # padding con ceros
       elif cur > exp_t:
           return X[..., :exp_t]  # truncar
   ```

6. **Escala a mV**
   ```python
   X = (X * 1e3).astype(np.float32)  # ✅ Convertir a mV
   ```

## 📈 Impacto Esperado

### Antes de la Corrección
- ❌ Predicciones inconsistentes
- ❌ Modelo recibía datos en forma/escala incorrecta
- ❌ Features no coincidían con el entrenamiento
- ❌ Accuracy efectivo ≈ random (50%)

### Después de la Corrección
- ✅ Predicciones consistentes con el entrenamiento
- ✅ Datos preprocesados idénticamente
- ✅ Features alineadas con los pesos del modelo
- ✅ Accuracy esperado según métricas de entrenamiento

## 🔬 Detalles Técnicos del Modelo

### Arquitectura EEGClassificationModel

```python
Input: (batch, 64 canales, 480 muestras)
  ↓
Conv1d(64 → 64, kernel=11, padding=5)
BatchNorm1d(64)
ReLU
  ↓
Conv1d(64 → 128, kernel=11, padding=5)
BatchNorm1d(128)
  ↓ (batch, 128, 480)
PositionalEncoding(128)
  ↓
TransformerBlock(128, heads=4)
TransformerBlock(128, heads=4)
  ↓ (batch, 128, 480)
MeanPooling(dim=-1)
  ↓ (batch, 128)
Linear(128 → 32)
ReLU + Dropout
Linear(32 → 1)
  ↓
Output: logit (batch, 1)
```

**Crucial:** El modelo espera exactamente:
- 64 canales en entrada
- 480 timesteps
- Sin filtros aplicados (entrenó con señal "cruda" a 160Hz)

## 🎯 Recomendaciones de Uso

### Para Inference
```bash
# Usar modelo entrenado con dataset correcto
python predecir.py archivo.edf left_rest_model.pt --json
```

### Para Nuevo Entrenamiento
Si necesitas re-entrenar:
1. ✅ Mantén `SAMPLE_RATE = 160`
2. ✅ Mantén `DURATION = 3.0` (480 muestras)
3. ✅ Usa 64 canales EEG
4. ✅ NO apliques filtros bandpass/notch
5. ✅ Convierte a mV (× 1e3)

### Variables de Entorno (opcional)
Si quieres sobrescribir desde fuera:
```bash
export EEG_CHANNELS=64
export EEG_SAMPLE_RATE=160
export EEG_DURATION=3.0
```

## 📝 Checklist de Validación

- [x] Frecuencia de muestreo: 160 Hz
- [x] Duración de épocas: 3 segundos (480 muestras)
- [x] Número de canales: 64
- [x] Filtro bandpass: DESHABILITADO
- [x] Filtro notch: DESHABILITADO
- [x] Escala a mV: HABILITADO (× 1e3)
- [x] Normalización de longitud: padding/truncate a 480
- [x] Clases: ["left", "right"]
- [x] Threshold de decisión: 0.5

## 🚨 Advertencia para Futuras Modificaciones

**NUNCA** modifiques estos parámetros sin re-entrenar el modelo completo:
- `n_channels`
- `T` (longitud temporal)
- `fs` (frecuencia de muestreo)
- Presencia de filtros (bandpass/notch)

Cambiar cualquiera de estos valores **invalidará el modelo** existente y requerirá entrenamiento desde cero.

---

**Fecha de corrección:** Noviembre 12, 2025  
**Archivos modificados:**
- `/ml-api/python/predecir.py` (parámetros por defecto en `predict_one_edf`)

**Estado:** ✅ CORREGIDO - Predicciones ahora alineadas con entrenamiento
