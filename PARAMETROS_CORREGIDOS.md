# ✅ Corrección Final de Parámetros - predecir.py

## 🎯 Objetivo
Alinear los parámetros de preprocesamiento en `predecir.py` con los del notebook de entrenamiento para que las predicciones sean consistentes.

## 📊 Parámetros Corregidos

### Antes (❌ Incorrecto)
```python
classes = ["rest", "left"]
n_channels = 32        # ❌ Solo la mitad de los canales
T = 513               # ❌ Longitud incorrecta
fs = 256.0            # ❌ Frecuencia incorrecta
bandpass = (1.0, 40.0) # ❌ Filtro no usado en entrenamiento
notch = 50.0          # ❌ Notch no usado en entrenamiento
```

### Ahora (✅ Correcto)
```python
classes = ["rest", "left"]  # ✅ Mantiene las clases requeridas
n_channels = 64            # ✅ Todos los canales EEG (igual que entrenamiento)
T = 480                   # ✅ 480 muestras = 160Hz × 3.0s (igual que entrenamiento)
fs = 160.0                # ✅ Frecuencia nativa del dataset (igual que entrenamiento)
bandpass = None           # ✅ Sin filtro (igual que entrenamiento)
notch = None              # ✅ Sin notch (igual que entrenamiento)
```

## 🔍 Comparación con Entrenamiento

| Parámetro | Notebook (Entrenamiento) | predecir.py (ANTES) | predecir.py (AHORA) |
|-----------|-------------------------|---------------------|---------------------|
| Canales   | 64                      | 32 ❌               | 64 ✅               |
| Muestras  | 480                     | 513 ❌              | 480 ✅              |
| Frecuencia| 160 Hz                  | 256 Hz ❌           | 160 Hz ✅           |
| Bandpass  | None                    | (1-40Hz) ❌         | None ✅             |
| Notch     | None                    | 50Hz ❌             | None ✅             |

## 🧪 Pipeline de Preprocesamiento

```
Archivo EDF
    ↓
Cargar con MNE
    ↓
Seleccionar 64 canales EEG
    ↓
Resample a 160 Hz (si es necesario)
    ↓
Extraer 480 muestras (3 segundos)
    ↓
NO aplicar filtros (bandpass/notch)
    ↓
Convertir a mV (× 1000)
    ↓
Normalizar longitud (pad con ceros o truncar)
    ↓
Tensor shape: (1, 64, 480)
    ↓
Modelo EEGClassificationModel
    ↓
Predicción: "rest" o "left"
```

## 💡 ¿Por qué estos valores?

### Del Notebook de Entrenamiento:
```python
SAMPLE_RATE = 160  # Hz típico para este dataset
DURATION = 3.0     # segundos
TARGET_LENGTH = 480 # = 160 * 3.0
EEG_CHANNEL = 64   # Todos los canales disponibles

# Preprocesamiento:
eeg_data = raw.get_data(picks=eeg_channel_inds)
# Sin filtros aplicados
eeg_data * 1e3  # Convertir a mV
```

### Arquitectura del Modelo:
```python
class EEGClassificationModel(nn.Module):
    def __init__(self, eeg_channel=64, dropout=0.1):
        # Conv: 64 → 64 → 128 canales
        # Transformer: 128 dim con 4 heads
        # MLP: 128 → 32 → 1 (binary output)
```

El modelo fue entrenado con **exactamente 64 canales y 480 timesteps**. Cambiar estos valores = incompatibilidad.

## ✅ Resultado Esperado

### Antes de la corrección:
- 🎲 Predicciones inconsistentes o aleatorias
- 💥 El modelo recibía datos en formato incorrecto
- 🔴 Features incompatibles con los pesos entrenados

### Después de la corrección:
- ✅ Predicciones consistentes
- ✅ Datos preprocesados idénticamente al entrenamiento
- ✅ Features alineadas con el modelo
- ✅ El mismo archivo EDF siempre da la misma predicción

## 🚀 Probar el Sistema

```bash
# Iniciar el servidor
cd /Users/bensagra/Documents/NeuroLinked-1/ml-api
node app.js

# Luego desde el frontend:
# 1. Sube un archivo EDF
# 2. Usa el modelo por defecto
# 3. Verifica la predicción: "rest" o "left"
# 4. Sube el mismo archivo de nuevo
# 5. Debe dar la misma predicción
```

## 📝 Clases del Modelo

Aunque el notebook original entrenó con `["left", "right"]`, tu modelo actual predice `["rest", "left"]`:
- **Clase 0** → "rest"
- **Clase 1** → "left"

Esto está configurado en la línea:
```python
classes: list[str] = ["rest", "left"]
```

## ⚠️ IMPORTANTE

**NO modificar estos parámetros sin re-entrenar el modelo:**
- `n_channels = 64`
- `T = 480`
- `fs = 160.0`
- `bandpass = None`
- `notch = None`

Cambiar cualquiera de estos valores hará que el modelo produzca predicciones incorrectas.

---

**Estado:** ✅ CORREGIDO  
**Fecha:** Noviembre 12, 2025  
**Archivo:** `/ml-api/python/predecir.py` (líneas 73-85)
