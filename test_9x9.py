#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para Testar Modelo LTN Sudoku 9x9
Projeto Final - Inteligência Artificial - UFAM
Equipe: Bianka Vasconcelos, Micael Viana, Vinicius Chagas
"""

import sys
import os
import argparse
from pathlib import Path

# Adicionar o diretório do projeto ao path
sys.path.append(str(Path(__file__).parent))

from main import test_model_for_dimension

def main():
    """
    Script principal para testar modelo 9x9
    """
    parser = argparse.ArgumentParser(description='Testar Modelo LTN Sudoku 9x9')
    parser.add_argument('--data-dir', type=str, default='data', help='Diretório dos dados')
    parser.add_argument('--model-path', type=str, default='models/sudoku_ltn_9x9.pth', 
                       help='Caminho para o modelo treinado')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TESTE MODELO LTN SUDOKU 9x9")
    print("Projeto Final - Inteligência Artificial - UFAM")
    print("Equipe: Bianka Vasconcelos, Micael Viana, Vinicius Chagas")
    print("=" * 80)
    
    print(f"\n🧪 INICIANDO TESTE 9x9")
    print(f"📁 Diretório de dados: {args.data_dir}")
    print(f"💾 Modelo: {args.model_path}")
    
    try:
        # Testar modelo 9x9
        test_model_for_dimension(9, args.model_path, args.data_dir)
        
        print(f"\n🎉 TESTE 9x9 CONCLUÍDO!")
        
    except Exception as e:
        print(f"\n❌ ERRO DURANTE O TESTE: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 