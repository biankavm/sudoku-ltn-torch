#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para Testar Modelo LTN Sudoku 4x4
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
    Script principal para testar modelo 4x4
    """
    parser = argparse.ArgumentParser(description='Testar Modelo LTN Sudoku 4x4')
    parser.add_argument('--data-dir', type=str, default='data', help='Diretório dos dados')
    parser.add_argument('--model-path', type=str, default='models/sudoku_ltn_4x4.pth', 
                       help='Caminho para o modelo treinado')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TESTE MODELO LTN SUDOKU 4x4")
    print("Projeto Final - Inteligência Artificial - UFAM")
    print("Equipe: Bianka Vasconcelos, Micael Viana, Vinicius Chagas")
    print("=" * 80)
    
    print(f"\n🧪 INICIANDO TESTE 4x4")
    print(f"📁 Diretório de dados: {args.data_dir}")
    print(f"💾 Modelo: {args.model_path}")
    
    try:
        # Testar modelo 4x4
        test_model_for_dimension(4, args.model_path, args.data_dir)
        
        print(f"\n🎉 TESTE 4x4 CONCLUÍDO!")
        
    except Exception as e:
        print(f"\n❌ ERRO DURANTE O TESTE: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 