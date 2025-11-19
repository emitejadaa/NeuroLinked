#!/usr/bin/env python3
"""
Script de validación para verificar que el preprocesamiento
está correctamente alineado entre entrenamiento y predicción.
"""

import sys
import os

# Colores para terminal
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_mark(condition):
    return f"{GREEN}✓{RESET}" if condition else f"{RED}✗{RESET}"

def test_preprocessing_params():
    """Verifica que los parámetros por defecto sean correctos."""
    
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Validación de Parámetros de Preprocesamiento{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    # Importar el módulo
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml-api', 'python'))
    
    try:
        # Leer el código fuente directamente para verificar defaults
        script_path = os.path.join(os.path.dirname(__file__), 'ml-api', 'python', 'predecir.py')
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Parámetros esperados (del entrenamiento)
        expected = {
            'n_channels': 64,
            'T': 480,
            'fs': 160.0,
            'classes': '["left", "right"]',
            'bandpass': 'None',
            'notch': 'None',
        }
        
        results = {}
        
        # Buscar cada parámetro en la firma de predict_one_edf
        import re
        
        # Extraer la función predict_one_edf (buscar con más flexibilidad)
        match = re.search(
            r'def predict_one_edf\(.*?\) -> dict:',
            content,
            re.DOTALL
        )
        
        if match:
            func_def = match.group(0)
            
            # Verificar cada parámetro
            results['n_channels'] = 'n_channels: int = 64' in func_def
            results['T'] = 'T: int = 480' in func_def
            results['fs'] = 'fs: float = 160.0' in func_def
            results['classes'] = 'classes: list[str] = ["left", "right"]' in func_def
            results['bandpass'] = 'bandpass: tuple[float, float] | None = None' in func_def
            results['notch'] = 'notch: float | None = None' in func_def
            
            print(f"📊 {YELLOW}Parámetros de Preprocesamiento:{RESET}\n")
            
            # Tabla de resultados
            print(f"{'Parámetro':<20} {'Esperado':<20} {'Estado':<10}")
            print(f"{'-'*50}")
            
            print(f"{'n_channels':<20} {'64':<20} {check_mark(results['n_channels']):<10}")
            print(f"{'T (muestras)':<20} {'480':<20} {check_mark(results['T']):<10}")
            print(f"{'fs (Hz)':<20} {'160.0':<20} {check_mark(results['fs']):<10}")
            print(f"{'classes':<20} {'[left, right]':<20} {check_mark(results['classes']):<10}")
            print(f"{'bandpass':<20} {'None':<20} {check_mark(results['bandpass']):<10}")
            print(f"{'notch':<20} {'None':<20} {check_mark(results['notch']):<10}")
            
            print()
            
            # Resumen
            all_correct = all(results.values())
            
            if all_correct:
                print(f"{GREEN}✅ TODOS LOS PARÁMETROS ESTÁN CORRECTOS{RESET}")
                print(f"\n{GREEN}El preprocesamiento está alineado con el entrenamiento.{RESET}")
                print(f"{GREEN}Las predicciones deberían ser consistentes ahora.{RESET}")
            else:
                print(f"{RED}❌ ALGUNOS PARÁMETROS SON INCORRECTOS{RESET}")
                incorrect = [k for k, v in results.items() if not v]
                print(f"\n{RED}Parámetros incorrectos: {', '.join(incorrect)}{RESET}")
                print(f"\n{YELLOW}⚠️  Revisa el archivo predecir.py y ajusta los valores por defecto.{RESET}")
                return False
            
            # Información adicional
            print(f"\n{BLUE}📝 Notas Importantes:{RESET}")
            print(f"  • Duración de época: {480/160.0:.1f} segundos")
            print(f"  • Sin filtros aplicados (bandpass/notch)")
            print(f"  • Usando todos los 64 canales EEG disponibles")
            print(f"  • Clasificación binaria: left vs right")
            
            return all_correct
        else:
            print(f"{RED}❌ No se pudo encontrar la función predict_one_edf{RESET}")
            return False
            
    except Exception as e:
        print(f"{RED}❌ Error durante la validación: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print(f"\n{BLUE}{'='*60}{RESET}\n")

if __name__ == "__main__":
    success = test_preprocessing_params()
    sys.exit(0 if success else 1)
