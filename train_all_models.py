#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para Treinar Todos os Modelos LTN (4x4 e 9x9)
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
    Script principal para treinar todos os modelos
    """
    parser = argparse.ArgumentParser(description='Treinar Todos os Modelos LTN Sudoku')
    parser.add_argument('--epochs-4x4', type=int, default=50, help='Número de épocas para modelo 4x4')
    parser.add_argument('--epochs-9x9', type=int, default=50, help='Número de épocas para modelo 9x9')
    parser.add_argument('--data-dir', type=str, default='data', help='Diretório dos dados')
    parser.add_argument('--skip-4x4', action='store_true', help='Pular treinamento 4x4')
    parser.add_argument('--skip-9x9', action='store_true', help='Pular treinamento 9x9')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TREINAMENTO COMPLETO - MODELOS LTN SUDOKU 4x4 E 9x9")
    print("Projeto Final - Inteligência Artificial - UFAM")
    print("Equipe: Bianka Vasconcelos, Micael Viana, Vinicius Chagas")
    print("=" * 80)
    
    start_time_total = time.time()
    results = {}
    
    # Treinar modelo 4x4
    if not args.skip_4x4:
        print(f"\n🎓 INICIANDO TREINAMENTO 4x4")
        print(f"📊 Épocas: {args.epochs_4x4}")
        print(f"📁 Diretório de dados: {args.data_dir}")
        
        try:
            model_path_4x4, training_results_4x4 = train_model_for_dimension(4, args.data_dir, args.epochs_4x4)
            results['4x4'] = {
                'success': True,
                'model_path': model_path_4x4,
                'training_results': training_results_4x4
            }
            print(f"✅ TREINAMENTO 4x4 CONCLUÍDO COM SUCESSO!")
            print(f"💾 Modelo salvo em: {model_path_4x4}")
        except Exception as e:
            print(f"❌ ERRO DURANTE O TREINAMENTO 4x4: {e}")
            results['4x4'] = {
                'success': False,
                'error': str(e)
            }
    else:
        print(f"\n⏭️  PULANDO TREINAMENTO 4x4")
        results['4x4'] = {'skipped': True}
    
    # Treinar modelo 9x9
    if not args.skip_9x9:
        print(f"\n🎓 INICIANDO TREINAMENTO 9x9")
        print(f"📊 Épocas: {args.epochs_9x9}")
        print(f"📁 Diretório de dados: {args.data_dir}")
        
        try:
            model_path_9x9, training_results_9x9 = train_model_for_dimension(9, args.data_dir, args.epochs_9x9)
            results['9x9'] = {
                'success': True,
                'model_path': model_path_9x9,
                'training_results': training_results_9x9
            }
            print(f"✅ TREINAMENTO 9x9 CONCLUÍDO COM SUCESSO!")
            print(f"💾 Modelo salvo em: {model_path_9x9}")
        except Exception as e:
            print(f"❌ ERRO DURANTE O TREINAMENTO 9x9: {e}")
            results['9x9'] = {
                'success': False,
                'error': str(e)
            }
    else:
        print(f"\n⏭️  PULANDO TREINAMENTO 9x9")
        results['9x9'] = {'skipped': True}
    
    # Resumo final
    total_time = time.time() - start_time_total
    
    print(f"\n" + "=" * 80)
    print("RESUMO FINAL DO TREINAMENTO")
    print("=" * 80)
    
    for dimension, result in results.items():
        print(f"\n📊 MODELO {dimension}:")
        if 'skipped' in result:
            print(f"  ⏭️  Pulado")
        elif result['success']:
            print(f"  ✅ Concluído com sucesso")
            print(f"  💾 Modelo: {result['model_path']}")
            
            # Estatísticas do treinamento
            training_results = result['training_results']
            total_samples = 0
            total_time_dim = 0
            
            for situation, res in training_results.items():
                if 'error' not in res:
                    total_samples += res['samples']
                    total_time_dim += res['training_time']
                    print(f"    📈 {situation}: {res['samples']} amostras, {res['training_time']:.2f}s")
                else:
                    print(f"    ❌ {situation}: Erro - {res['error']}")
            
            print(f"    📈 Total: {total_samples} amostras, {total_time_dim:.2f}s")
        else:
            print(f"  ❌ Falhou: {result['error']}")
    
    print(f"\n⏱️  TEMPO TOTAL: {total_time:.2f}s")
    
    # Verificar se ambos os modelos foram treinados com sucesso
    success_count = sum(1 for r in results.values() if r.get('success', False))
    total_attempted = sum(1 for r in results.values() if 'skipped' not in r)
    
    if success_count == total_attempted:
        print(f"\n🎉 TODOS OS MODELOS TREINADOS COM SUCESSO!")
        print(f"💡 Próximos passos:")
        print(f"  # Testar modelo 4x4:")
        print(f"  python test_4x4.py")
        print(f"  # Testar modelo 9x9:")
        print(f"  python test_9x9.py")
        print(f"  # Demonstração completa:")
        print(f"  python demo_separated_models.py")
        return 0
    else:
        print(f"\n⚠️  {success_count}/{total_attempted} modelos treinados com sucesso")
        return 1

if __name__ == "__main__":
    exit(main()) 