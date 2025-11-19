#!/usr/bin/env python3
"""
Script de prueba para verificar que las predicciones sean deterministas
(mismo EDF → misma predicción cada vez)
"""

import sys
import os

# Agregar el path del módulo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml-api', 'python'))

def test_determinism():
    """Prueba que el mismo archivo EDF da la misma predicción múltiples veces."""
    
    # Verificar que existe un archivo EDF de prueba
    edf_dir = "ml-api/uploads/edf"
    if not os.path.exists(edf_dir):
        print(f"❌ No existe el directorio {edf_dir}")
        return False
    
    # Buscar el primer archivo EDF
    edf_files = [f for f in os.listdir(edf_dir) if f.endswith('.edf')]
    
    if not edf_files:
        print(f"❌ No se encontraron archivos EDF en {edf_dir}")
        print("   Por favor sube un archivo EDF desde el frontend primero")
        return False
    
    edf_path = os.path.join(edf_dir, edf_files[0])
    model_path = "ml-api/python/left_rest_model_20251022_142504.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ No existe el modelo {model_path}")
        return False
    
    print(f"🧪 Probando determinismo con: {edf_files[0]}")
    print(f"🔬 Modelo: {model_path}\n")
    
    # Importar la función de predicción
    from predecir import predict_one_edf
    
    # Hacer múltiples predicciones
    predictions = []
    probabilities = []
    
    print("Ejecutando 5 predicciones consecutivas...\n")
    
    for i in range(5):
        try:
            result = predict_one_edf(edf_path, model_path)
            pred = result['pred']
            prob = result['prob']
            
            predictions.append(pred)
            probabilities.append(prob)
            
            print(f"  Predicción {i+1}: {pred:6s} (prob: {prob:.6f})")
            
        except Exception as e:
            print(f"❌ Error en predicción {i+1}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print()
    
    # Verificar consistencia
    all_same_pred = all(p == predictions[0] for p in predictions)
    all_same_prob = all(abs(p - probabilities[0]) < 1e-8 for p in probabilities)
    
    if all_same_pred and all_same_prob:
        print(f"✅ ¡ÉXITO! Todas las predicciones son idénticas")
        print(f"   Predicción: {predictions[0]}")
        print(f"   Probabilidad: {probabilities[0]:.8f}")
        print(f"\n🎯 El modelo es DETERMINISTA")
        return True
    else:
        print(f"❌ FALLO: Las predicciones son diferentes")
        print(f"\n📊 Predicciones:")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            print(f"   {i+1}. {pred:6s} (prob: {prob:.8f})")
        
        # Verificar qué está variando
        if not all_same_pred:
            print(f"\n⚠️  Las CLASES predichas varían")
        if not all_same_prob:
            print(f"⚠️  Las PROBABILIDADES varían")
            diffs = [abs(p - probabilities[0]) for p in probabilities[1:]]
            print(f"   Diferencia máxima: {max(diffs):.10f}")
        
        return False

if __name__ == "__main__":
    print("="*60)
    print("Test de Determinismo - NeuroLinked")
    print("="*60 + "\n")
    
    success = test_determinism()
    
    print("\n" + "="*60)
    
    sys.exit(0 if success else 1)
