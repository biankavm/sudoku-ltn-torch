#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstração dos Modelos LTN Separados para Sudoku 4x4 e 9x9
Projeto Final - Inteligência Artificial - UFAM
Equipe: Bianka Vasconcelos, Micael Viana, Vinicius Chagas

Este script demonstra como usar os modelos treinados separadamente
para cada dimensão de Sudoku.
"""

import sys
import os
import argparse
from pathlib import Path

# Adicionar o diretório do projeto ao path
sys.path.append(str(Path(__file__).parent))

from core.sudoku_board import SudokuBoard
from solver.ltn_solver import SudokuLTNSolver
from main import create_sample_4x4_sudoku, create_sample_sudoku, create_unsolvable_4x4_sudoku

def demo_4x4_model():
    """
    Demonstra o uso do modelo 4x4
    """
    print("\n🎲 DEMONSTRAÇÃO MODELO 4x4")
    print("=" * 50)
    
    # Inicializar solver 4x4
    solver_4x4 = SudokuLTNSolver(board_size=4)
    
    # Carregar modelo se existir
    model_path_4x4 = "models/sudoku_ltn_4x4.pth"
    if os.path.exists(model_path_4x4):
        solver_4x4.load_model(model_path_4x4)
        print(f"✅ Modelo 4x4 carregado: {model_path_4x4}")
    else:
        print(f"⚠️  Modelo 4x4 não encontrado: {model_path_4x4}")
        print("Executando sem modelo pré-treinado...")
    
    # Testar com sudoku solucionável
    print(f"\n📋 Testando Sudoku 4x4 Solucionável:")
    board_4x4_solvable = create_sample_4x4_sudoku()
    print(board_4x4_solvable)
    
    resultado_4x4 = solver_4x4.solve_sudoku(board_4x4_solvable)
    print(f"🎯 Resultado: {resultado_4x4['sucesso']}")
    print(f"📝 Motivo: {resultado_4x4['motivo']}")
    print(f"🔄 Iterações: {resultado_4x4['iteracoes']}")
    
    if resultado_4x4['sucesso']:
        print(f"\n📋 Sudoku 4x4 resolvido:")
        print(resultado_4x4['board_final'])
    
    # Testar com sudoku impossível
    print(f"\n📋 Testando Sudoku 4x4 Impossível:")
    board_4x4_impossible = create_unsolvable_4x4_sudoku()
    print(board_4x4_impossible)
    
    resultado_4x4_impossible = solver_4x4.solve_sudoku(board_4x4_impossible)
    print(f"🎯 Resultado: {resultado_4x4_impossible['sucesso']}")
    print(f"📝 Motivo: {resultado_4x4_impossible['motivo']}")

def demo_9x9_model():
    """
    Demonstra o uso do modelo 9x9
    """
    print("\n🎲 DEMONSTRAÇÃO MODELO 9x9")
    print("=" * 50)
    
    # Inicializar solver 9x9
    solver_9x9 = SudokuLTNSolver(board_size=9)
    
    # Carregar modelo se existir
    model_path_9x9 = "models/sudoku_ltn_9x9.pth"
    if os.path.exists(model_path_9x9):
        solver_9x9.load_model(model_path_9x9)
        print(f"✅ Modelo 9x9 carregado: {model_path_9x9}")
    else:
        print(f"⚠️  Modelo 9x9 não encontrado: {model_path_9x9}")
        print("Executando sem modelo pré-treinado...")
    
    # Testar com sudoku 9x9
    print(f"\n📋 Testando Sudoku 9x9:")
    board_9x9 = create_sample_sudoku()
    print(board_9x9)
    
    resultado_9x9 = solver_9x9.solve_sudoku(board_9x9)
    print(f"🎯 Resultado: {resultado_9x9['sucesso']}")
    print(f"📝 Motivo: {resultado_9x9['motivo']}")
    print(f"🔄 Iterações: {resultado_9x9['iteracoes']}")
    
    if resultado_9x9['sucesso']:
        print(f"\n📋 Sudoku 9x9 resolvido:")
        print(resultado_9x9['board_final'])

def compare_models():
    """
    Compara o desempenho dos modelos 4x4 e 9x9
    """
    print("\n📊 COMPARAÇÃO DOS MODELOS")
    print("=" * 50)
    
    # Verificar se os modelos existem
    model_4x4_exists = os.path.exists("models/sudoku_ltn_4x4.pth")
    model_9x9_exists = os.path.exists("models/sudoku_ltn_9x9.pth")
    
    print(f"📁 Modelo 4x4: {'✅ Disponível' if model_4x4_exists else '❌ Não encontrado'}")
    print(f"📁 Modelo 9x9: {'✅ Disponível' if model_9x9_exists else '❌ Não encontrado'}")
    
    if model_4x4_exists:
        # Informações do modelo 4x4
        size_4x4 = os.path.getsize("models/sudoku_ltn_4x4.pth") / (1024 * 1024)  # MB
        print(f"💾 Tamanho modelo 4x4: {size_4x4:.2f} MB")
    
    if model_9x9_exists:
        # Informações do modelo 9x9
        size_9x9 = os.path.getsize("models/sudoku_ltn_9x9.pth") / (1024 * 1024)  # MB
        print(f"💾 Tamanho modelo 9x9: {size_9x9:.2f} MB")
    
    # Verificar relatórios de treinamento
    report_4x4_exists = os.path.exists("models/training_report_4x4.txt")
    report_9x9_exists = os.path.exists("models/training_report_9x9.txt")
    
    print(f"📄 Relatório 4x4: {'✅ Disponível' if report_4x4_exists else '❌ Não encontrado'}")
    print(f"📄 Relatório 9x9: {'✅ Disponível' if report_9x9_exists else '❌ Não encontrado'}")

def main():
    """
    Função principal da demonstração
    """
    parser = argparse.ArgumentParser(description='Demonstração dos Modelos LTN Separados')
    parser.add_argument('--demo-4x4', action='store_true', help='Demonstrar modelo 4x4')
    parser.add_argument('--demo-9x9', action='store_true', help='Demonstrar modelo 9x9')
    parser.add_argument('--compare', action='store_true', help='Comparar modelos')
    parser.add_argument('--all', action='store_true', help='Executar todas as demonstrações')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DEMONSTRAÇÃO MODELOS LTN SEPARADOS - SUDOKU 4x4 E 9x9")
    print("Projeto Final - Inteligência Artificial - UFAM")
    print("Equipe: Bianka Vasconcelos, Micael Viana, Vinicius Chagas")
    print("=" * 80)
    
    if args.all or args.demo_4x4:
        demo_4x4_model()
    
    if args.all or args.demo_9x9:
        demo_9x9_model()
    
    if args.all or args.compare:
        compare_models()
    
    # Se nenhum argumento foi fornecido, executar todas as demonstrações
    if not any([args.demo_4x4, args.demo_9x9, args.compare, args.all]):
        print("\n🎯 EXECUTANDO TODAS AS DEMONSTRAÇÕES")
        demo_4x4_model()
        demo_9x9_model()
        compare_models()
    
    print(f"\n🎉 DEMONSTRAÇÃO CONCLUÍDA!")
    print(f"\n💡 DICAS DE USO:")
    print(f"  # Treinar modelo 4x4:")
    print(f"  python train_4x4.py --epochs 50")
    print(f"  # Treinar modelo 9x9:")
    print(f"  python train_9x9.py --epochs 50")
    print(f"  # Testar modelo 4x4:")
    print(f"  python test_4x4.py")
    print(f"  # Testar modelo 9x9:")
    print(f"  python test_9x9.py")

if __name__ == "__main__":
    main() 