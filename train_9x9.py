#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para Treinar Modelo LTN Sudoku 9x9
Projeto Final - Inteligência Artificial - UFAM
Equipe: Bianka Vasconcelos, Micael Viana, Vinicius Chagas
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Adicionar o diretório do projeto ao path
sys.path.append(str(Path(__file__).parent))

from main import train_model_for_dimension

def main():
    """
    Script principal para treinar modelo 9x9
    """
    parser = argparse.ArgumentParser(description='Treinar Modelo LTN Sudoku 9x9')
    parser.add_argument('--epochs', type=int, default=50, help='Número de épocas de treinamento')
    parser.add_argument('--data-dir', type=str, default='data', help='Diretório dos dados')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TREINAMENTO MODELO LTN SUDOKU 9x9")
    print("Projeto Final - Inteligência Artificial - UFAM")
    print("Equipe: Bianka Vasconcelos, Micael Viana, Vinicius Chagas")
    print("=" * 80)
    
    print(f"\n🎓 INICIANDO TREINAMENTO 9x9")
    print(f"📊 Épocas: {args.epochs}")
    print(f"📁 Diretório de dados: {args.data_dir}")
    
    try:
        # Treinar modelo 9x9
        model_path, training_results = train_model_for_dimension(9, args.data_dir, args.epochs)
        
        print(f"\n🎉 TREINAMENTO 9x9 CONCLUÍDO COM SUCESSO!")
        print(f"💾 Modelo salvo em: {model_path}")
        
        # Resumo dos resultados
        print(f"\n📊 RESUMO DO TREINAMENTO:")
        total_samples = 0
        total_time = 0
        
        for situation, results in training_results.items():
            if 'error' not in results:
                print(f"  ✅ {situation}: {results['samples']} amostras, {results['training_time']:.2f}s")
                total_samples += results['samples']
                total_time += results['training_time']
            else:
                print(f"  ❌ {situation}: Erro - {results['error']}")
        
        print(f"\n📈 ESTATÍSTICAS GERAIS:")
        print(f"  Total de amostras: {total_samples}")
        print(f"  Tempo total: {total_time:.2f}s")
        print(f"  Modelo: {model_path}")
        
    except Exception as e:
        print(f"\n❌ ERRO DURANTE O TREINAMENTO: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 